import math

import torch


class LangevinDynamicsCosineAnnealingScheduler:
    def __init__(
        self,
        min_sigma: float,
        max_sigma: float,
    ) -> None:
        self.max_sigma = max_sigma
        self.min_sigma = max(min_sigma, 0.000001)

    def _get_sigma(
        self,
        mcmc_step: int,
        num_mcmc_steps: int,
        device: torch.device
    ) -> float:
        """Retrieve the current sigma for Langevin Dynamics"""
        return (
            self.min_sigma + 0.5*(self.max_sigma - self.min_sigma)
            * (1 + math.cos(math.pi * mcmc_step / num_mcmc_steps))
        )

    def _get_ld_noise(
        self,
        mcmc_step: int,
        num_mcmc_steps: int,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve the current sigma for Langevin Dynamics"""
        langevian_dynamcs = torch.randn_like(action, device=action.device)
        sigma = self._get_sigma(mcmc_step, num_mcmc_steps)

        return langevian_dynamcs * sigma

    def apply_noise(
        self,
        mcmc_step: int,
        num_mcmc_steps: int,
        action: torch.Tensor
    ) -> torch.Tensor:
        ld_noise = self._get_ld_noise(mcmc_step, num_mcmc_steps, action)

        return action + ld_noise
