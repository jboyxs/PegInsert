"""Capture the final policy history before Isaac Lab resets timed-out environments."""

import copy

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationManager


class PegInsetEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        # The PegInset critic reads the policy group. A history-free reader gives
        # us its final frame without advancing any surviving environment's history.
        frame_cfg = copy.deepcopy(self.cfg.observations.policy)
        frame_cfg.history_length = 0
        frame_cfg.concatenate_terms = False
        self._frame_reader = ObservationManager({"policy": frame_cfg}, self)

    def step(self, action):
        self.extras["timeout_bootstrap"] = None
        return super().step(action)

    def _reset_idx(self, env_ids):
        if not self.cfg.is_finite_horizon and hasattr(self, "reset_time_outs"):
            eligible = self.reset_time_outs & ~self.reset_terminated
            if isinstance(env_ids, slice):
                env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
            ids = env_ids[eligible[env_ids]].long()
            if ids.numel():
                self.extras["timeout_bootstrap"] = (ids, self._final_policy_history(ids))
        super()._reset_idx(env_ids)

    def _final_policy_history(self, ids):
        frames = self._frame_reader.compute_group("policy")
        frames = [frame[ids].flatten(1) for frame in frames.values()]
        group = self.cfg.observations.policy
        length = group.history_length
        if not length:
            return torch.cat(frames, dim=1)
        previous = self.obs_buf["policy"][ids]
        if group.history_order == "time":
            frame = torch.cat(frames, dim=1)
            return torch.cat((previous[:, frame.shape[1]:], frame), dim=1)
        # Default layout stores all five frames of each term contiguously.
        chunks = previous.split([length * frame.shape[1] for frame in frames], dim=1)
        histories = [
            torch.cat((old[:, frame.shape[1]:], frame), dim=1)
            for old, frame in zip(chunks, frames, strict=True)
        ]
        return torch.cat(histories, dim=1)
