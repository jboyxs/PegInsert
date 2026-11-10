"""Potential shaping regressions through the reward manager, without Isaac Sim."""

import copy
from types import SimpleNamespace
import unittest

import torch

from isaaclab.managers import RewardManager

from peginsert.tasks.pegact.config.franka.env_cfg import PegInsetEnvCfg


class PotentialRewardTests(unittest.TestCase):
    def setUp(self):
        cfg = PegInsetEnvCfg()
        self.positions = torch.tensor([[0.03, 0.0, 0.12], [0.03, 0.0, 0.12]])
        quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)

        def asset(positions):
            return SimpleNamespace(data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=positions),
                root_quat_w=SimpleNamespace(torch=quaternions.clone()),
            ))

        self.env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            cfg=cfg,
            sim=SimpleNamespace(is_playing=lambda: True),
            step_dt=cfg.sim.dt * cfg.decimation,
            max_episode_length_s=cfg.episode_length_s,
            scene={"held_asset": asset(self.positions), "fixed_asset": asset(torch.zeros_like(self.positions))},
            extras={},
        )
        context_cfg = copy.deepcopy(cfg.terminations.progress_context)
        self.context = context_cfg.func(context_cfg, self.env)
        context_cfg.func = self.context
        self.env.termination_manager = SimpleNamespace(
            get_term_cfg=lambda name: context_cfg,
            terminated=torch.zeros(2, dtype=torch.bool),
            time_outs=torch.zeros(2, dtype=torch.bool),
        )
        self.manager = RewardManager({"assembly_progress": cfg.rewards.assembly_progress}, self.env)
        self.manager.reset()

    def step(self, positions):
        self.positions[:] = torch.tensor(positions)
        self.context(self.env, **self.env.cfg.terminations.progress_context.params)
        return self.manager.compute(self.env.step_dt).clone()

    def test_approach_alignment_insertion_and_retreat(self):
        for position in ([0.01, 0.0, 0.02], [0.0, 0.0, 0.009], [0.0, 0.0, 0.004]):
            self.assertTrue((self.step([position, position]) > 0.0).all())
        self.assertTrue((self.step([[0.0, 0.0, 0.009]] * 2) < 0.0).all())
        # With gamma < 1, hovering has a small negative shaping reward.
        hover = self.step([[0.0, 0.0, 0.009]] * 2)
        self.assertTrue(((hover < 0.0) & (hover > -0.011)).all())

    def test_lowering_beside_hole_has_no_insertion_bonus(self):
        self.positions[:] = torch.tensor([[0.04, 0.0, 0.009]] * 2)
        self.manager.reset()
        self.assertTrue((self.step([[0.04, 0.0, 0.0]] * 2) < 0.0).all())

    def test_terminal_discounted_return_is_independent_of_path_and_duration(self):
        gamma = self.env.cfg.rewards.assembly_progress.params["gamma"]

        def rollout(path, success):
            self.positions[:] = torch.tensor([[0.03, 0.0, 0.12]] * 2)
            self.env.termination_manager.terminated.zero_()
            self.manager.reset()
            total = torch.zeros(2)
            for index, position in enumerate(path):
                if index == len(path) - 1:
                    self.env.termination_manager.terminated.fill_(True)
                total += gamma ** index * self.step([position, position])
            self.assertEqual(bool(self.context.is_success.all()), success)
            return total

        success_return = rollout([[0.0, 0.0, 0.009], [0.0, 0.0, 0.0]], success=True)
        failure_return = rollout(
            [[0.02, 0.0, 0.02], [0.0, 0.0, 0.009], [0.03, 0.0, 0.1], [0.04, 0.0, 0.2]],
            success=False,
        )
        torch.testing.assert_close(success_return, failure_return)

    def test_partial_reset_uses_new_poses_and_preserves_other_history(self):
        self.step([[0.0, 0.0, 0.004]] * 2)
        self.positions[0] = torch.tensor([0.03, 0.0, 0.12])
        # Cached context errors still contain the old close-to-success pose.
        self.manager.reset(env_ids=torch.tensor([0]))
        reward = self.step(self.positions.tolist())
        self.assertTrue(((reward < 0.0) & (reward > -0.011)).all())
        self.assertGreater(reward[0].item(), reward[1].item())

    def test_true_terminal_and_timeout_bootstrap(self):
        self.env.termination_manager.terminated[0] = True
        self.env.termination_manager.time_outs.fill_(True)
        reward = self.step([[0.0, 0.0, 0.004]] * 2)
        self.assertLess(reward[0].item(), 0.0)
        self.assertGreater(reward[1].item(), 0.0)

        self.env.cfg.is_finite_horizon = True
        self.positions[:] = torch.tensor([[0.03, 0.0, 0.12]] * 2)
        self.manager.reset()
        finite_reward = self.step([[0.0, 0.0, 0.004]] * 2)
        torch.testing.assert_close(finite_reward, reward[0].expand(2))

    def test_reward_manager_step_dt_does_not_change_shaping_magnitude(self):
        reward = self.step([[0.0, 0.0, 0.009]] * 2)
        self.positions[:] = torch.tensor([[0.03, 0.0, 0.12]] * 2)
        self.manager.reset()
        self.env.step_dt /= 2
        torch.testing.assert_close(self.step([[0.0, 0.0, 0.009]] * 2), reward)


if __name__ == "__main__":
    unittest.main()
