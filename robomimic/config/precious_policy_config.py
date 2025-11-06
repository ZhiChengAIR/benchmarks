"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig


class PreciousPolicyConfig(BaseConfig):
    ALGO_NAME = "precious_policy"

    def train_config(self):
        """
        Setting up training parameters for EBT Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(PreciousPolicyConfig, self).train_config()

        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon
        self.train.hdf5_filter_key = "train"
        self.train.hdf5_validation_filter_key = "valid"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # Transformer parameters
        self.algo.transformer.enabled = True
        self.algo.transformer.embed_dim = 256
        self.algo.transformer.num_layers = 8
        self.algo.transformer.num_heads = 4
        self.algo.transformer.attn_dropout = 0.3
        self.algo.transformer.proj_dropout = 0.0

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        self.algo.precious.randomize_mcmc_step_size_scale = 1
        self.algo.precious.scale_alpha_with_energy_temp = 1
        self.algo.precious.randomize_mcmc_num_steps = 3
        self.algo.precious.mcmc_num_steps = 3
        self.algo.precious.mcmc_step_size = 10000
        self.algo.precious.clamp_future_grads = False
        self.algo.precious.clamp_futures_grad_max_change = 9.0
        self.algo.precious.mcmc_step_size_learnable = False
        self.algo.precious.max_sigma = 0.3
        self.algo.precious.min_sigma = 0.001
        self.algo.precious.truncate_mcmc = False
        self.algo.precious.min_grad = 0.002
        self.algo.precious.max_mcmc_steps = 20
        self.algo.precious.mu = 0.9
        self.algo.precious.inf_sigma = 0.1
        self.algo.precious.add_inf_ld_noise = False
