"""Compare pre-reset histories and PPO targets with independent successor observations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from peginsert.tasks.pegact.config.franka.agents.rsl_rl_ppo_cfg import PegInsetPPORunnerCfg
from peginsert.tasks.pegact.env import PegInsetEnv


@configclass
class HistoryCfg(ObservationGroupCfg):
    position = ObservationTermCfg(func=lambda env: env.state[:, :2])
    command = ObservationTermCfg(func=lambda env: env.state[:, 2:])


class TimeoutBootstrapTests(unittest.TestCase):
    def make_env(self, order="term"):
        group = HistoryCfg(history_length=5, history_order=order)
        env = object.__new__(PegInsetEnv)
        env._is_closed = True
        env.cfg = SimpleNamespace(observations=SimpleNamespace(policy=group), is_finite_horizon=False)
        env.scene = SimpleNamespace(num_envs=4)
        env.sim = SimpleNamespace(device="cpu", is_playing=lambda: True)
        env.state = torch.arange(12).float().reshape(4, 3)
        env.extras = {}
        with patch.object(ManagerBasedRLEnv, "load_managers"):
            env.load_managers()
        env.observation_manager = ObservationManager({"policy": group}, env)
        for _ in range(5):
            env.obs_buf = env.observation_manager.compute(update_history=True)
            env.state += 1
        return env

    def test_preview_matches_real_history_append_without_changing_live_history(self):
        for order in ("term", "time"):
            with self.subTest(order=order):
                env = self.make_env(order)
                ids = torch.tensor([0, 2])
                preview = env._final_policy_history(ids)
                torch.testing.assert_close(env.observation_manager.compute()["policy"], env.obs_buf["policy"])
                expected = env.observation_manager.compute(update_history=True)["policy"][ids]
                torch.testing.assert_close(preview, expected)

    @torch.no_grad()
    def test_rollout_bootstraps_only_pure_timeout_from_pre_reset_successor(self):
        env = self.make_env()
        obs = TensorDict(env.obs_buf, batch_size=[4])
        cfg = handle_deprecated_rsl_rl_cfg(PegInsetPPORunnerCfg(), check_rsl_rl_version()).to_dict()
        cfg.update(num_steps_per_env=1, multi_gpu=None)
        alg = resolve_callable(cfg["algorithm"]["class_name"]).construct_algorithm(
            obs, SimpleNamespace(num_envs=4, num_actions=1), cfg, "cpu"
        )
        alg.act(obs)
        prior_values = alg.transition.values.clone()
        # Continuing, timeout, terminal, and simultaneous timeout + terminal.
        env.reset_time_outs = torch.tensor([False, True, False, True])
        env.reset_terminated = torch.tensor([False, False, True, True])
        dones = env.reset_time_outs | env.reset_terminated
        with patch.object(ManagerBasedRLEnv, "_reset_idx"):
            env._reset_idx(torch.where(dones)[0])
        ids, _ = env.extras["timeout_bootstrap"]
        self.assertEqual(ids.tolist(), [1])
        successor = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[4])
        final_values = alg.critic(successor).squeeze(-1)
        post_reset = successor.clone()
        post_reset["policy"][dones] = -100
        rewards = torch.ones(4)
        extras = dict(env.extras, time_outs=env.reset_time_outs)
        alg.process_env_step(post_reset, rewards, dones, extras)
        alg.compute_returns(post_reset)
        expected = torch.ones(4)
        expected[1] += alg.gamma * final_values[1]
        torch.testing.assert_close(alg.storage.rewards[0, :, 0], expected)
        expected[0] += alg.gamma * final_values[0]
        torch.testing.assert_close(alg.storage.returns[0, :, 0], expected)
        torch.testing.assert_close(alg.storage.values[0], prior_values)
        torch.testing.assert_close(rewards, torch.ones(4))
