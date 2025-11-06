from typing import Optional
import math

import torch


class LangevinDynamicsCosineAnnealingScheduler:
    def __init__(
        self,
        min_sigma: float,
        max_sigma: float,
        inf_sigma: float,
        add_inf_ld_noise: bool
    ) -> None:
        self.max_sigma = max_sigma
        self.min_sigma = max(min_sigma, 0.000001)
        self.inf_sigma = inf_sigma
        self.add_inf_ld_noise = add_inf_ld_noise

    def _get_sigma(
        self,
        mcmc_step: int,
        num_mcmc_steps: int,
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
        action: torch.Tensor,
        inference_mode: Optional[bool] = False
    ) -> torch.Tensor:
        """Retrieve the current sigma for Langevin Dynamics"""
        langevian_dynamcs = torch.randn_like(action, device=action.device)
        if inference_mode and self.add_inf_ld_noise:
            sigma = self.inf_sigma
        elif not inference_mode:
            assert num_mcmc_steps is not None, "num_mcmc_steps is None"
            sigma = self._get_sigma(mcmc_step, num_mcmc_steps)
        else:
            sigma = 0

        return langevian_dynamcs * sigma

    def apply_noise(
        self,
        mcmc_step: int,
        action: torch.Tensor,
        num_mcmc_steps: Optional[int] = None,
        inference_mode: bool = False
    ) -> torch.Tensor:
        ld_noise = self._get_ld_noise(
            mcmc_step=mcmc_step,
            num_mcmc_steps=num_mcmc_steps,
            action=action,
            inference_mode=inference_mode
        )

        return action + ld_noise
