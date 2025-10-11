""" This file contains nets used for Diffusion Policy. """
import math
from typing import Union

import torch
import torch.nn as nn

from robomimic.models.transformers import SpatioTemporalEncoder, DiT


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ObsTemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_dropout,
        proj_dropout,
        num_layers,
        num_heads,
        n_obs_steps
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        # input embedding stem
        self.encoder = SpatioTemporalEncoder(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            output_dim=embed_dim,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
            n_obs_steps=n_obs_steps
        )

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        x = self.input_emb(x) + self.input_pos_emb
        x = self.encoder(x, self.mask)

        return x


class EBTTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        embed_dim,
        output_dim,
        attn_dropout,
        proj_dropout,
        num_layers,
        num_heads,
        n_obs_steps,
        horizon
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, embed_dim)
        self.input_pos_emb = nn.Parameter(torch.zeros(1, horizon, embed_dim))
        self.time_emb = SinusoidalPosEmb(embed_dim)

        self.decoder = DiT(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            output_dim=output_dim,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout
        )
        mask = torch.tril(torch.ones(horizon, n_obs_steps)).view(
            1, 1, horizon, n_obs_steps
        )
        self.register_buffer("mask", mask)

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        x = self.input_emb(sample) + self.input_pos_emb
        t = self.time_emb(timesteps)[:, None]
        c = cond
        x = self.decoder(x=x, c=c, t=t, mask=None, memory_mask=self.mask)

        # (B,T,C)
        return x
