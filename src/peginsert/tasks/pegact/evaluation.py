"""Evaluation configuration and complete-episode accounting for trained policies."""

import torch


def make_evaluation_env_cfg(*, num_envs=64, device="cuda:0", seed=12345,
                            state_table_size=512, physics="newton_mjwarp", headless=False):
    """Use the original play-mode air grasps, uniform sampling, and full gravity."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    if num_envs < 1 or state_table_size < 1:
        raise ValueError("num_envs and state_table_size must be positive")
    cfg, agent_cfg = resolve_task_config(
        "PegInset", "rsl_rl_cfg_entry_point", play_mode=True, overrides=[f"physics={physics}"],
    )
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.events.variable_gravity = None
    cfg.curriculum.gravity_adr = None
    cfg.curriculum.difficulty_scheduler = None
    reset = cfg.events.reset_strategies.params
    reset["state_table_size"] = state_table_size
    reset["report"] = False
    # play_mode already selects only grasp_asset_in_air and makes both samplers uniform.
    # Preserve training's pose noise, IK settings, and collision acceptance conditions.
    if headless:
        cfg.sim.visualizer_cfgs = []
    elif physics == "newton_mjwarp":
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=(0.9, 0.8, 0.6), lookat=(0.45, 0.0, 0.08), max_visible_envs=4,
        )]
    cfg.validate()
    return cfg, agent_cfg


class EpisodeStatistics:
    """Assign fixed episode quotas so fast-ending envs cannot dominate the result.

    All envs start at step zero. Completed episodes after an env has filled its
    quota are ignored while the remaining envs finish their assigned episodes.
    """

    def __init__(self, num_envs, episodes, device="cpu"):
        if num_envs < 1 or episodes < num_envs:
            raise ValueError("Require episodes >= num_envs >= 1")
        self.target = episodes
        self.quotas = torch.full((num_envs,), episodes // num_envs, device=device, dtype=torch.long)
        self.quotas[:episodes % num_envs] += 1
        self.counts = torch.zeros_like(self.quotas)
        self.lengths = torch.zeros_like(self.quotas)
        self.returns = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.completed = self.successes = self.failures = self.timeouts = 0
        self.total_length = 0
        self.total_return = 0.0
        self.termination_counts = {}

    def update(self, rewards, dones, term_flags, timeout_terms, initial_slots):
        """Read per-step terminal flags, which survive the environment's auto-reset."""
        active = self.counts < self.quotas
        self.lengths += active.long()
        self.returns += torch.where(active, rewards.double(), 0.0)
        ids = (dones.bool() & active).nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return []
        names = list(term_flags)
        flags = torch.stack([term_flags[name][ids].bool() for name in names], dim=1).cpu().tolist()
        rows = zip(ids.tolist(), self.lengths[ids].tolist(), self.returns[ids].tolist(),
                   initial_slots[ids].tolist(), flags, strict=True)
        records = []
        for env_id, length, episode_return, slot, values in rows:
            fired = [name for name, value in zip(names, values, strict=True) if value]
            failure = any(name != "success" and name not in timeout_terms for name in fired)
            success = "success" in fired and not failure
            timeout = any(name in timeout_terms for name in fired)
            outcome = "success" if success else "failure" if failure else "timeout" if timeout else "failure"
            self.completed += 1
            self.successes += int(outcome == "success")
            self.failures += int(outcome == "failure")
            self.timeouts += int(outcome == "timeout")
            self.total_length += length
            self.total_return += episode_return
            for name in fired:
                self.termination_counts[name] = self.termination_counts.get(name, 0) + 1
            records.append({
                "episode_id": self.completed - 1, "env_id": env_id, "initial_state_id": slot,
                "outcome": outcome, "success": success, "timeout": timeout,
                "termination_terms": fired, "length": length, "return": episode_return,
            })
        self.counts[ids] += 1
        self.lengths[ids] = 0
        self.returns[ids] = 0
        return records

    def summary(self):
        return {
            "requested_episodes": self.target, "completed_episodes": self.completed,
            "successful_episodes": self.successes, "failed_episodes": self.failures,
            "timeout_episodes": self.timeouts,
            "success_rate": self.successes / self.completed if self.completed else None,
            "mean_episode_length": self.total_length / self.completed if self.completed else None,
            "mean_episode_return": self.total_return / self.completed if self.completed else None,
            "termination_counts": self.termination_counts.copy(),
            "discarded_partial_episodes": int(((self.counts < self.quotas) & (self.lengths > 0)).sum()),
            "complete": self.completed == self.target,
        }
