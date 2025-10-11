"""
Contains torch Modules for observation tokenisers
such as encoders (e.g. LowDimTokeniser, VisionTokeniser, ...)
"""
import textwrap
from typing import Dict, Any, Sequence
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Lambda, Compose

import robomimic.models.base_nets as BaseNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

# NOTE: this is required for the backbone classes to be found by the `eval` call in the core networks
from robomimic.models.base_nets import (
    Module,
    Sequential,
    MLP,
    RNN_Base,
    ResNet18Conv,
    SpatialSoftmax,
    FeatureAggregator
)
from robomimic.models.obs_core import (
    EncoderCore,
    VisualCore,
    VisualCoreLanguageConditioned
)

def obs_tokeniser_factory(
    obs_shapes,
    output_dim,
    feature_activation=nn.ReLU,
    encoder_kwargs=None
):
    """
    Utility function to create an @ObservationTokeniser from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = ObservationTokeniser(
        feature_activation=feature_activation,
        output_dim=output_dim
    )
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality]) if encoder_kwargs is None else \
            deepcopy(encoder_kwargs[obs_modality])

        # Sanity check for kwargs in case they don't exist / are None
        if enc_kwargs.get("core_kwargs", None) is None:
            enc_kwargs["core_kwargs"] = {}
        # Add in input shape info
        enc_kwargs["core_kwargs"]["input_shape"] = obs_shape
        # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
        if enc_kwargs["core_class"] is not None:
            enc_kwargs["core_kwargs"] = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_ENCODER_CORES[enc_kwargs["core_class"]],
                dic=enc_kwargs["core_kwargs"],
                copy=False,
            )

        # Add in input shape info
        randomizers = []
        obs_randomizer_class_list = enc_kwargs["obs_randomizer_class"]
        obs_randomizer_kwargs_list = enc_kwargs["obs_randomizer_kwargs"]

        if not isinstance(obs_randomizer_class_list, list):
            obs_randomizer_class_list = [obs_randomizer_class_list]

        if not isinstance(obs_randomizer_kwargs_list, list):
            obs_randomizer_kwargs_list = [obs_randomizer_kwargs_list]

        rand_input_shape = obs_shape
        for rand_class, rand_kwargs in zip(obs_randomizer_class_list, obs_randomizer_kwargs_list):
            rand = None
            if rand_class is not None:
                rand_kwargs["input_shape"] = rand_input_shape
                rand_kwargs = extract_class_init_kwargs_from_dict(
                    cls=ObsUtils.OBS_RANDOMIZERS[rand_class],
                    dic=rand_kwargs,
                    copy=False,
                )
                rand = ObsUtils.OBS_RANDOMIZERS[rand_class](**rand_kwargs)
                rand_input_shape = rand.output_shape_in(rand_input_shape)
            randomizers.append(rand)

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizers=randomizers,
        )

    enc.make()
    return enc


class Patchify(BaseNets.Module):
    def __init__(self):
        super(Patchify, self).__init__()
        self.flatten = BaseNets.Flatten(start_dim=-2, end_dim=-1)
        self.transpose = BaseNets.Transpose(dim1=1, dim2=2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.transpose(x)

        return x

    def output_shape(self, input_dim):
        output_shape = deepcopy(input_dim)
        output_shape = self.flatten.output_shape(output_shape)
        output_shape = self.transpose.output_shape(output_shape)

        return output_shape


class ObservationGroupTokeniser(BaseNets.Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationTokeniser object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """

    def __init__(
        self,
        observation_group_shapes: Dict[str, Any],
        output_dim: int,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ObservationGroupTokeniser, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all([isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes])

        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_tokeniser_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
                output_dim=output_dim
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """
        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(
                self.nets[obs_group].forward(inputs[obs_group])
            )

        return torch.cat(outputs, dim=1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.observation_group_shapes:
            msg += '\n'
            indent = ' ' * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationTokeniser(BaseNets.Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks.
    """

    def __init__(
        self,
        output_dim: int,
        feature_activation: nn.Module = nn.ReLU,
    ):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.
        """
        super(ObservationTokeniser, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self._locked = False
        self.lowdim_shape_dict = OrderedDict()
        self.output_dim = output_dim

    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizers=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert not self._locked, "ObservationTokeniser: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationTokeniser: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(net, BaseNets.Module), "ObservationTokeniser: @net must be instance of Module class"
            assert (net_class is None) and (net_kwargs is None) and (share_net_from is None), \
                "ObservationTokeniser: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        randomizers = [] if randomizers is None else randomizers # handles None
        if not isinstance(randomizers, list): # handle single randomizer
            randomizers = [randomizers]
        rand_output_shape = shape
        for rand in randomizers:
            if rand is not None:
                assert isinstance(rand, BaseNets.Randomizer)
                rand_output_shape = rand.output_shape_in(rand_output_shape)
        if net_kwargs is not None:
            net_kwargs["input_shape"] = rand_output_shape

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = nn.ModuleList(randomizers)
        self.obs_share_mods[name] = share_net_from

        if ObsUtils.OBS_KEYS_TO_MODALITIES[name] == "low_dim":
            self.lowdim_shape_dict[name] = net_kwargs["input_shape"]

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](
                    **self.obs_nets_kwargs[k], feature_dimension=self.output_dim
                )
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.lowdim_tokeniser = LowDimTokeniser(
            lowdim_shape_dict=self.lowdim_shape_dict,
            output_dim=self.output_dim,
        )

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def _get_vis_lang_lowdim_info(self):
        """
        Helper function to extract information on vision, language and lowdim keys.
        """

        # get the indices that correspond to RGB and lang
        rgb_inds = []
        rgb_inds_need_lang_cond = []
        lang_inds = []
        lang_keys = []
        lowdim_inds = []
        for ind, k in enumerate(self.obs_shapes):
            if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                rgb_inds.append(ind)
                if (self.obs_nets[k] is not None) and isinstance(self.obs_nets[k], VisualCoreLanguageConditioned):
                    rgb_inds_need_lang_cond.append(ind)
            elif k == LangUtils.LANG_EMB_OBS_KEY:
                lang_inds.append(ind)
                lang_keys.append(k)
            elif ObsUtils.key_is_obs_modality(key=k, obs_modality="low_dim"):
                lowdim_inds.append(ind)
        assert len(lang_inds) <= 1

        # whether language features should be included in network features
        include_lang_feat = True
        if (len(rgb_inds_need_lang_cond) > 0):
            include_lang_feat = False
        return rgb_inds, rgb_inds_need_lang_cond, lang_inds, lang_keys, include_lang_feat, lowdim_inds

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        rgb_inds, rgb_inds_need_lang_cond, lang_inds, lang_keys, include_lang_feat, lowdim_inds = self._get_vis_lang_lowdim_info()

        # process modalities by order given by @self.obs_shapes
        feats = []
        lowdim_dict = {}
        for ind, k in enumerate(self.obs_shapes):
            # maybe skip language input
            if (not include_lang_feat) and (ind in lang_inds):
                continue
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    x = rand.forward_in(x)
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                if (ind in rgb_inds_need_lang_cond):
                    x = self.obs_nets[k](x, lang_emb=obs_dict[lang_keys[0]])
                else:
                    x = self.obs_nets[k](x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            for rand in reversed(self.obs_randomizers[k]):
                if rand is not None:
                    x = rand.forward_out(x)
            if ind in lowdim_inds:
                lowdim_dict[k] = x
            else:
                feats.append(x)

        lowdim_feats = self.lowdim_tokeniser(lowdim_dict)
        feats = [lowdim_feats[:, None]] + feats

        # concatenate all features together
        return torch.cat(feats, dim=1)


class LowDimTokeniser(EncoderCore):
    """
    Encodes all lowdim data into a single embedding.
    """

    def __init__(
        self,
        lowdim_shape_dict: Dict,
        output_dim: int,
        dropout: float = 0.0
    ):
        input_shape = [sum(
            shape[0] for shape in lowdim_shape_dict.values()
        )]
        super(LowDimTokeniser, self).__init__(input_shape=input_shape)
        self.input_dict = lowdim_shape_dict
        print(self.input_dict)
        self.input_shape = input_shape
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.lowdim_encoder = BaseNets.MLP(
            input_dim=self.input_shape[0],
            output_dim=output_dim,
            layer_dims=(output_dim,),
            activation=approx_gelu,
            dropouts=(dropout,),
            normalization=True,
            output_activation=approx_gelu
        )

    def forward(
        self,
        lowdim_dict: Dict
    ):
        lowdim_keys = sorted(lowdim_dict.keys())
        lowdim = torch.cat([lowdim_dict[key] for key in lowdim_keys], dim=-1)
        lowdim_emb = self.lowdim_encoder(lowdim)

        return lowdim_emb

    def output_shape(self, input_shape: int = None):
        return self.lowdim_encoder.output_shape()

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nencoder_net={}".format(self.lowdim_encoder), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class VisionTokeniser(VisualCore):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        dropout: float = 0.0,
        backbone_class: str = "ResNet18Conv",
        backbone_kwargs: Dict[str, Any] = None,
        patchify: bool = True,
        feature_dimension: int = 64,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network. Defaults
                to "ResNet18Conv".
            backbone_kwargs (dict): kwargs for the visual backbone network (optional)
            flatten (bool): whether to flatten the visual features
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisionTokeniser, self).__init__(input_shape=input_shape)
        self.patchify = patchify

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_ENCODER_BACKBONES[backbone_class],
                dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, BaseNets.ConvBase)

        net_list = [self.backbone]

        if self.patchify:
            self.patchify_module = Patchify()
            net_list.append(self.patchify_module)

        # get input dim after tokenisation
        self.feature_dimension = feature_dimension
        token_shape = self.backbone.output_shape(input_shape)
        if self.patchify:
            token_shape = self.patchify_module.output_shape(token_shape)
        input_dim = token_shape[-1]

        def approx_gelu(): return nn.GELU(approximate="tanh")
        print("input_dim:", input_dim)
        print()
        print()
        self.projector = BaseNets.MLP(
            input_dim=input_dim,
            output_dim=feature_dimension,
            layer_dims=[feature_dimension],
            activation=approx_gelu,
            dropouts=[dropout],
            normalization=True,
            output_activation=approx_gelu
        )

        net_list.append(self.projector)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        feat_shape = self.backbone.output_shape(input_shape)
        if self.patchify:
            feat_shape = self.patchify_module.output_shape(feat_shape)
        feat_shape[-1] = self.feature_dimension
        return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisionTokeniser, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\nPatchify={}".format(self.patchify_module), indent)
        msg += textwrap.indent("\nProjector={}".format(self.projector), indent)
        msg = header + '(' + msg + '\n)'

        return msg
