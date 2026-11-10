"""Configuration and geometry checks that do not require Isaac Sim."""

import copy
import importlib.metadata
import unittest

import gymnasium as gym
import torch

import peginsert  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config

from peginsert.tasks.pegact.assembly_profile import AssemblyProfile
from peginsert.tasks.pegact.config.franka.task_cfg import ASSEMBLY_PROFILE


class PegInsetTests(unittest.TestCase):
    def test_cli_discovery(self):
        entries = importlib.metadata.entry_points(group="isaaclab.tasks")
        entry = next(entry for entry in entries if entry.name == "peginsert")
        self.assertEqual(entry.load().__name__, "peginsert.tasks")
        spec = gym.spec("PegInset")
        self.assertTrue(spec.kwargs["env_cfg_entry_point"].endswith(":PegInsetEnvCfg"))
        self.assertEqual(spec.kwargs["default_agent"], "rsl_rl")

    def test_all_backends_and_play_mode(self):
        for backend in ("isaacsim_physx", "physx", "newton_mjwarp"):
            for play in (False, True):
                with self.subTest(backend=backend, play=play):
                    cfg, agent = resolve_task_config(
                        "PegInset", "rsl_rl_cfg_entry_point", play_mode=play,
                        overrides=[f"physics={backend}"],
                    )
                    cfg.validate()
                    self.assertTrue(cfg.scene.fixed_asset.spawn.usd_path.endswith("/rectangular_hole_4mm.usd"))
                    self.assertTrue(cfg.scene.held_asset.spawn.usd_path.endswith("/rectangular_peg_4mm.usd"))
                    self.assertEqual(agent.experiment_name, "peg_inset")
                    self.assertEqual(cfg.rewards.assembly_progress.params["gamma"], agent.algorithm.gamma)
                    reset = cfg.events.reset_strategies.params
                    self.assertEqual(reset["reset_assets"], ["nistboard", "fixed_asset", "held_asset", "robot"])
                    choices = reset["reset_term"].params["terms"]["reset_strategies"].params["terms"]
                    self.assertEqual(len(choices), 1 if play else 3)
                    self.assertEqual(reset["state_table_size"], 512 if play else 32768)
                    self.assertEqual(cfg.scene.num_envs, 128 if play else 4096)
                    if play:
                        self.assertIsNone(cfg.events.variable_gravity)
                        self.assertIsNone(cfg.curriculum.gravity_adr)

    def test_configs_do_not_share_mutable_reset_parameters(self):
        first, _ = resolve_task_config("PegInset", None, overrides=[])
        first.events.reset_strategies.params["reset_assets"].clear()
        first.terminations.progress_context.params["assembly_profile"].segments.clear()
        second, _ = resolve_task_config("PegInset", None, overrides=[])
        self.assertEqual(len(second.events.reset_strategies.params["reset_assets"]), 4)
        self.assertEqual(len(second.terminations.progress_context.params["assembly_profile"].segments), 2)
        self.assertEqual(len(ASSEMBLY_PROFILE.segments), 2)

    def test_assembly_profile_samples_square_peg_geometry(self):
        profile = AssemblyProfile(copy.deepcopy(ASSEMBLY_PROFILE))
        for fraction, height in ((0.0, 0.0), (0.7, 0.009)):
            pos, quat = profile.sample((fraction, fraction), 32, "cpu")
            expected = torch.tensor([0.0, 0.0, height]).expand(32, -1)
            torch.testing.assert_close(pos, expected)
            torch.testing.assert_close(torch.linalg.vector_norm(quat, dim=-1), torch.ones(32))
            # The square peg permits only multiples of 90 degrees about z in the hole.
            yaw = 2 * torch.atan2(quat[:, 2], quat[:, 3])
            torch.testing.assert_close(torch.sin(2 * yaw), torch.zeros(32), atol=1e-6, rtol=0)
        pos, quat = profile.sample((0.7, 1.5), 128, "cpu")
        self.assertTrue(torch.isfinite(pos).all())
        self.assertTrue(torch.isfinite(quat).all())
        self.assertTrue((pos[:, :2].abs() <= 0.01).all())
        self.assertTrue(((pos[:, 2] >= 0.01) & (pos[:, 2] <= 0.015)).all())


if __name__ == "__main__":
    unittest.main()
