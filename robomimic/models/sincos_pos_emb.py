from typing import Optional

import torch


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: int,
    temporal_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim: The embedding dimension of inputs. It must be divisible
            by 16.
        spatial_size: The spatial dimension of positional embeddings. If an
            integer is provided, the same size is applied to both spatial
            dimensions (height and width).
        temporal_size: The temporal dimension of postional embeddings
            (number of frames).
        device: The device that the pos emb should be sent to.

    Returns:
        The 3D sinusoidal positional embeddings of shape
        [temporal_size, spatial_size[0] * spatial_size[1], embed_dim].
    """
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32)
    grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # 2. Temporal
    grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[None, :, :]

    pos_embed_spatial = pos_embed_spatial.repeat_interleave(
        temporal_size, dim=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = pos_embed_temporal.repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )  # [T, H*W, D // 4]

    pos_embed = torch.concat(
        [pos_embed_temporal, pos_embed_spatial], dim=-1
    )  # [T, H*W, D]

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim: The embedding dimension.
        grid: Grid of positions with shape (H * W,).

    Returns:
        The 2D sinusoidal positional embeddings with shape (H * W, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim: The embedding dimension D
        pos: 1D tensor of positions with shape (M,)

    Returns:
        Sinusoidal positional embeddings of shape (M, D).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(
        embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb
