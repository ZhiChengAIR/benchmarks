from typing import Optional
import torch


def generate_attention_mask(
    obs_tokens: torch.Tensor,
    n_obs_steps: int,
    action_tokens: Optional[torch.Tensor] = None,
    cross_attention: Optional[bool] = False
) -> torch.Tensor:
    """
    Generates a 4D attention mask for transformer attention.

    This function constructs a mask that supports both self-attention and
    cross-attention.

    Args:
        obs_tokens: The tokens relating to observation data
        action_tokens: Optional tensor of shape (batch, horizon, ...) used to
            determine the temporal extent for cross-attention.
        cross_attention: If True, enables cross-attention mode (e.g., decoder
            attending to encoder tokens).

    Returns:
        A boolean tensor of shape (batch, 1, total_tokens, total_tokens)
        representing the attention mask. A value of True means "masked" (i.e.,
        no attention allowed).
    """
    assert obs_tokens.shape[1] % n_obs_steps == 0, "mismatch in observation " + \
        "shape expectations"
    horizon = action_tokens.shape[1] if cross_attention else None
    device = obs_tokens.device
    batch_size = obs_tokens.shape[0]
    tokens_per_time_step = obs_tokens.shape[1] // n_obs_steps

    width = tokens_per_time_step * n_obs_steps
    height = horizon if cross_attention else width

    rows, cols = torch.arange(height).unsqueeze(1), torch.arange(width)
    mask = rows >= (cols // tokens_per_time_step)

    mask = mask[None, None].repeat(batch_size, 1, 1, 1).to(device)
    return mask
