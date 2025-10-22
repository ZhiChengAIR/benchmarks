"""
Implementation of EBT Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union, Sequence, Optional, List
import random
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
from robomimic.models.base_nets import RMSNorm
import robomimic.models.ebt_policy_nets as EBTNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.token_utils as TokUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("ebt_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return EBTPolicy, {}


class EBTPolicy(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        obs_dim = obs_encoder.output_shape()[0]
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        obs_temporal_encoder = EBTNets.ObsTemporalEncoder(
            input_dim=obs_dim,
            embed_dim=self.algo_config.transformer.embed_dim,
            num_layers=self.algo_config.transformer.num_layers,
            num_heads=self.algo_config.transformer.num_heads,
            n_obs_steps=self.algo_config.horizon.observation_horizon,
            attn_dropout=self.algo_config.transformer.attn_dropout,
            proj_dropout=self.algo_config.transformer.proj_dropout
        )

        # create network object
        energy_pred_net = EBTNets.EBTTransformer(
            input_dim=self.ac_dim,
            cond_dim=self.algo_config.transformer.embed_dim,
            embed_dim=self.algo_config.transformer.embed_dim,
            output_dim=self.ac_dim,
            num_layers=self.algo_config.transformer.num_layers,
            num_heads=self.algo_config.transformer.num_heads,
            attn_dropout=self.algo_config.transformer.attn_dropout,
            proj_dropout=self.algo_config.transformer.proj_dropout,
            n_obs_steps=self.algo_config.horizon.observation_horizon,
            horizon=self.algo_config.horizon.prediction_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "obs_temporal_encoder": obs_temporal_encoder,
                "energy_pred_net": energy_pred_net
            })
        })

        nets = nets.float().to(self.device)

        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        # set attrs
        self.nets = nets
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

        self.scale_alpha_with_energy_temp = self.algo_config.ebt.scale_alpha_with_energy_temp
        self.randomize_mcmc_num_steps = self.algo_config.ebt.randomize_mcmc_num_steps
        self.mcmc_step_size = self.algo_config.ebt.mcmc_step_size
        self.alpha = nn.Parameter(
            torch.tensor(
                float(self.mcmc_step_size),
                device=self.device
            ),
            requires_grad=self.algo_config.ebt.mcmc_step_size_learnable,
        )
        self.clamp_futures_grad = self.algo_config.ebt.clamp_future_grads
        self.clamp_futures_grad_max_change = self.algo_config.ebt.clamp_futures_grad_max_change
        self.mcmc_num_steps = self.algo_config.ebt.mcmc_num_steps
        self.truncate_mcmc = self.algo_config.ebt.truncate_mcmc
        self.langevin_dynamics_noise_std = torch.tensor(self.algo_config.ebt.langevin_dynamics_noise_std)
        self.ebl_norm = RMSNorm(self.ac_dim)
        self.randomize_mcmc_step_size_scale = self.algo_config.ebt.randomize_mcmc_step_size_scale
        self.max_mcmc_steps = self.algo_config.ebt.max_mcmc_steps
        self.min_grad = self.algo_config.ebt.min_grad

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for EBT Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(EBTPolicy, self).train_on_batch(batch, epoch, validate=validate)
            action = batch["actions"]

            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]
            obs_cond = self.nets["policy"]["obs_temporal_encoder"](
                obs_features
            )

            # sample noise to add to actions
            pred_action = torch.randn(action.shape, device=self.device)

            memory_mask = TokUtils.generate_attention_mask(
                obs_tokens=obs_cond,
                n_obs_steps=To,
                action_tokens=action,
                cross_attention=True
            )

            predicted_traj_list = []
            predicted_energies_list = []
            langevin_dynamics_noise_std = torch.clamp(
                self.langevin_dynamics_noise_std, min=0.000001
            )
            num_mcmc_steps = self._compute_num_mcmc_steps(
                no_randomness=False
            )
            grad_norms = []

            # Set to true for validation since grad would be off.
            with torch.set_grad_enabled(True):
                for i in range(num_mcmc_steps):
                    pred_action, pred_grad = self._energy_step(
                        trajectory=pred_action,
                        cond_tokens=obs_cond,
                        memory_mask=memory_mask,
                        final_stop=(i >= num_mcmc_steps - 1),
                        inference_mode=False,
                        langevin_dynamics_noise_std=langevin_dynamics_noise_std,
                        predicted_energies_list=predicted_energies_list,
                        predicted_traj_list=predicted_traj_list,
                    )
                    pred_grad_norm = pred_grad.norm(dim=(-1, -2))
                    grad_norms.append(pred_grad_norm)

            loss_info, loss = compute_loss(
                action,
                predicted_traj_list,
                predicted_energies_list,
                self.truncate_mcmc
            )
            info["losses"] = loss_info

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                    max_grad_norm=1.0
                )

                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)

                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info

    def _energy_step(
        self,
        trajectory: torch.Tensor,
        cond_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        inference_mode: bool,
        final_stop: bool,
        langevin_dynamics_noise_std: Optional[torch.Tensor] = None,
        predicted_energies_list: Optional[List[torch.Tensor]] = None,
        predicted_traj_list: Optional[List[torch.Tensor]] = None
    ):
        B = trajectory.shape[0]
        trajectory = trajectory.detach().requires_grad_()
        if final_stop:
            trajectory = self.ebl_norm(trajectory)

        if not inference_mode and self.langevin_dynamics_noise_std != 0:
            ld_noise = torch.randn_like(
                trajectory,
                device=trajectory.device,
            ) * langevin_dynamics_noise_std
            trajectory = trajectory + ld_noise

        energy_pred = self.nets["policy"]["energy_pred_net"](
            sample=trajectory,
            cond=cond_tokens,
            memory_mask=memory_mask
        )
        energy_pred = energy_pred.mean(dim=(-1, -2))

        alpha = self._compute_alpha(
            energy_pred=energy_pred,
            no_randomness=inference_mode,
            batch_size=B
        )

        predicted_traj_grad = self._compute_grad(
            energy_pred=energy_pred,
            trajectory=trajectory,
            final_stop=final_stop,
            create_graph=(not inference_mode)
        )

        trajectory = trajectory - alpha * predicted_traj_grad

        if not inference_mode:
            predicted_energies_list.append(energy_pred)
            predicted_traj_list.append(trajectory)

        return trajectory, predicted_traj_grad

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(EBTPolicy, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]

        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)

            # put actions into the queue
            self.action_queue.extend(action_sequence[0])

        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()

        # [1,Da]
        action = action.unsqueeze(0)
        return action

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim

        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        obs_cond = nets["policy"]["obs_temporal_encoder"](
            obs_features,
        )

        # initialize action from Guassian noise
        action_pred = torch.randn(
            (B, Tp, action_dim), device=self.device)

        memory_mask = TokUtils.generate_attention_mask(
            obs_tokens=obs_cond,
            n_obs_steps=To,
            action_tokens=action_pred,
            cross_attention=True
        )
        # Set to true for validation since grad would be off.
        with torch.set_grad_enabled(True):
            i = 0
            grad_pred_norm = float("inf")
            grad_norms = []
            while i < self.max_mcmc_steps - 1 and grad_pred_norm > self.min_grad:
                action_pred, grad_pred = self._energy_step(
                    trajectory=action_pred,
                    cond_tokens=obs_cond,
                    memory_mask=memory_mask,
                    inference_mode=True,
                    final_stop=False
                )
                grad_pred_norm = grad_pred.norm().detach().item()
                grad_norms.append(grad_pred_norm)
                i += 1
            action_pred, _ = self._energy_step(
                trajectory=action_pred,
                cond_tokens=obs_cond,
                memory_mask=memory_mask,
                final_stop=True,
                inference_mode=True
            )
        print("num steps:", i+1)

        start = To - 1
        end = start + Ta
        action_pred = action_pred[:, start:end]    # slice the window to execute now

        return action_pred

    def _compute_alpha(
        self,
        energy_pred: torch.Tensor,
        no_randomness: bool,
        batch_size: int,
    ) -> float:
        alpha = torch.clamp(self.alpha, min=0.0001)
        expanded_alpha = alpha.expand(batch_size, 1, 1)
        if not no_randomness and self.randomize_mcmc_step_size_scale != 1:
            scale = self.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            expanded_alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        exponentiated_energies = torch.exp(
            energy_pred[:, None, None]
        ) / self.scale_alpha_with_energy_temp
        energy_scaled_alpha = expanded_alpha * exponentiated_energies

        return energy_scaled_alpha

    def _compute_num_mcmc_steps(self, no_randomness: bool) -> int:
        if no_randomness or self.randomize_mcmc_num_steps <= 0:
            return self.mcmc_num_steps

        random_steps = random.randint(0, self.randomize_mcmc_num_steps + 1)
        return (self.mcmc_num_steps) + random_steps

    def _compute_grad(
        self,
        energy_pred: torch.Tensor,
        trajectory: torch.Tensor,
        final_stop: bool,
        create_graph
    ):
        if self.truncate_mcmc and final_stop:
            predicted_traj_grad = torch.autograd.grad(
                outputs=energy_pred.sum(),
                retain_graph=True,
                inputs=trajectory,
                create_graph=create_graph
            )[0]
        elif self.truncate_mcmc:
            predicted_traj_grad = torch.autograd.grad(
                outputs=energy_pred.sum(),
                inputs=trajectory,
                create_graph=False
            )[0]
        else:
            predicted_traj_grad = torch.autograd.grad(
                outputs=energy_pred.sum(),
                inputs=trajectory,
                create_graph=create_graph
            )[0]

        if self.clamp_futures_grad:
            min_and_max = self.clamp_futures_grad_max_change / (self.alpha)
            predicted_traj_grad = torch.clamp(
                predicted_traj_grad,
                min=-min_and_max,
                max=min_and_max
            )
        if torch.isnan(predicted_traj_grad).any() \
           or torch.isinf(predicted_traj_grad).any():
            raise ValueError("NaN or Inf gradients detected during MCMC.")

        return predicted_traj_grad

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def compute_loss(
    actions: torch.Tensor,
    predicted_traj_list: Sequence[torch.Tensor],
    predicted_energies_list: Sequence[torch.Tensor],
    truncate_mcmc: bool
):
    reconstruction_loss = 0
    total_mcmc_steps = len(predicted_energies_list)
    for mcmc_step, (predicted_embeddings, predicted_energy) in enumerate(zip(predicted_traj_list, predicted_energies_list)):
        if truncate_mcmc:
            if mcmc_step == (total_mcmc_steps - 1):
                reconstruction_loss = F.mse_loss(
                    predicted_embeddings, actions
                )
                final_reconstruction_loss = reconstruction_loss.detach()
        else:
            # loss calculations
            reconstruction_loss += F.mse_loss(
                predicted_embeddings,
                actions
            )
            if mcmc_step == (total_mcmc_steps - 1):
                final_reconstruction_loss = F.mse_loss(
                    predicted_embeddings,
                    actions
                ).detach()
                reconstruction_loss = reconstruction_loss / total_mcmc_steps

        # pure logging things (no function for training)
        if mcmc_step == 0:
            initial_loss = F.mse_loss(predicted_embeddings, actions).detach()
            initial_pred_energies = predicted_energy.squeeze().mean().detach()
        if mcmc_step == (total_mcmc_steps - 1):
            final_pred_energies = predicted_energy.squeeze().mean().detach()

    initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
    total_loss = reconstruction_loss

    info = {
        "init_loss": initial_loss,
        "final_step_loss": final_reconstruction_loss,
        "init_final_pred_energies_gap": initial_final_pred_energies_gap,
        "l2_loss": total_loss
    }

    return info, total_loss


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
