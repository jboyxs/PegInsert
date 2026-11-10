# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory reward terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-5000, 5000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-5000, 5000)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)


class assembly_potential_reward(ManagerTermBase):
    """Discounted potential shaping for approaching, aligning and inserting the peg.

    Returns ``(gamma * Phi(next) - Phi(previous)) / step_dt`` so RewardManager's
    time scaling leaves the configured weight as the shaping multiplier.
    True terminal states have zero potential. Infinite-horizon timeouts retain
    their geometric potential for value bootstrapping. Gamma must match the learner.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not 0.0 < cfg.params["gamma"] <= 1.0:
            raise ValueError("Potential shaping gamma must lie in (0, 1].")
        for name in ("hole_height", "approach_scale", "xy_scale", "tilt_scale"):
            if cfg.params[name] <= 0.0:
                raise ValueError(f"Potential shaping {name} must be positive.")
        self._context = env.termination_manager.get_term_cfg(cfg.params["context"]).func
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)

    def _potential(
        self, xy_distance: torch.Tensor, z_distance: torch.Tensor, tilt: torch.Tensor
    ) -> torch.Tensor:
        params = self.cfg.params
        above_hole = (z_distance - params["hole_height"]).clamp_min(0.0)
        distance = torch.hypot(xy_distance, above_hole)
        approach = 1.0 / (1.0 + distance / params["approach_scale"])
        alignment = torch.exp(
            -(xy_distance / params["xy_scale"]).square() - (tilt / params["tilt_scale"]).square()
        )
        insertion = alignment * (1.0 - z_distance / params["hole_height"]).clamp(0.0, 1.0)
        return 0.30 * approach + 0.25 * alignment + 0.45 * insertion

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset events run before RewardManager.reset. Read the new poses, since
        # the context's cached errors still describe the preceding terminal step.
        indices = slice(None) if env_ids is None else env_ids
        potential = self._potential(*self._context.alignment_errors())
        self._previous_potential[indices] = potential[indices]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gamma: float,
        hole_height: float,
        approach_scale: float,
        xy_scale: float,
        tilt_scale: float,
        context: str = "progress_context",
    ) -> torch.Tensor:
        potential = self._potential(
            self._context.xy_distance, self._context.z_distance, self._context.euler_xy_diff
        )
        terminal = env.termination_manager.terminated
        if env.cfg.is_finite_horizon:
            terminal = terminal | env.termination_manager.time_outs
        next_potential = torch.where(terminal, 0.0, potential)
        reward = (gamma * next_potential - self._previous_potential) / env.step_dt
        self._previous_potential.copy_(potential)
        return reward
