"""Check evaluation distribution and asynchronous episode accounting without simulation."""

import unittest

import torch

import peginsert  # noqa: F401
from isaaclab.app import validate_launch
from isaaclab_tasks.utils.hydra import resolve_task_config
from peginsert.tasks.pegact.evaluation import EpisodeStatistics, make_evaluation_env_cfg


class EvaluationTests(unittest.TestCase):
    def test_full_gravity_air_only_preserves_training_grasp_noise(self):
        train, _ = resolve_task_config("PegInset", None, overrides=["physics=newton_mjwarp"])
        for backend in ("newton_mjwarp", "isaacsim_physx", "physx"):
            cfg, _ = make_evaluation_env_cfg(num_envs=3, state_table_size=7, seed=73, physics=backend, headless=True)
            self.assertEqual(cfg.sim.gravity, (0., 0., -9.81))
            self.assertEqual(cfg.scene.num_envs, 3)
            self.assertEqual(cfg.seed, 73)
            self.assertIsNone(cfg.events.variable_gravity)
            self.assertIsNone(cfg.curriculum.gravity_adr)
            self.assertIsNone(cfg.curriculum.difficulty_scheduler)
            reset = cfg.events.reset_strategies.params
            train_reset = train.events.reset_strategies.params
            self.assertEqual(reset["state_table_size"], 7)
            self.assertEqual(set(reset["acceptance_conditions"]), set(train_reset["acceptance_conditions"]))
            choice = reset["reset_term"].params["terms"]["reset_strategies"].params
            train_choice = train_reset["reset_term"].params["terms"]["reset_strategies"].params
            self.assertEqual(list(choice["terms"]), ["grasp_asset_in_air"])
            for name, term in choice["terms"]["grasp_asset_in_air"].params["terms"].items():
                self.assertEqual(term.params, train_choice["terms"]["grasp_asset_in_air"].params["terms"][name].params)
            for sampler in (reset["sampling"], choice["sampling"]):
                self.assertEqual(type(sampler.strategies[0]).__name__, "UniformSamplingStrategyCfg")
                self.assertEqual(sampler.eps, 0.)
            self.assertEqual(cfg.terminations.progress_context.params["success_threshold"], 0.001)
            if backend == "newton_mjwarp":
                _, kit = validate_launch(cfg, {"headless": True, "livestream": 0})
                self.assertFalse(kit)

    def test_success_failure_overlap_and_timeout_are_disjoint_outcomes(self):
        stats = EpisodeStatistics(5, 5)
        flags = {
            "success": torch.tensor([True, True, False, False, True]),
            "abnormal": torch.tensor([False, True, False, False, False]),
            "oob": torch.tensor([False, False, False, True, False]),
            "time_out": torch.tensor([False, False, True, False, True]),
        }
        records = stats.update(torch.ones(5), torch.ones(5), flags, {"time_out"}, torch.arange(10, 15))
        self.assertEqual([r["outcome"] for r in records], ["success", "failure", "timeout", "failure", "success"])
        self.assertEqual(records[1]["termination_terms"], ["success", "abnormal"])
        self.assertEqual(records[2]["initial_state_id"], 12)
        summary = stats.summary()
        self.assertEqual(summary["success_rate"], 0.4)
        self.assertEqual(summary["failed_episodes"], 2)
        self.assertEqual(summary["timeout_episodes"], 1)
        self.assertTrue(summary["complete"])

    def test_quotas_wait_for_slow_env_and_reset_episode_accumulators(self):
        stats = EpisodeStatistics(2, 3)  # env 0: two episodes; env 1: one.
        slots = torch.tensor([4, 9])
        for step in range(4):
            dones = torch.tensor([step in (1, 3), True])
            flags = {"success": dones, "time_out": torch.zeros(2, dtype=torch.bool)}
            records = stats.update(torch.tensor([2., 3.]), dones, flags, {"time_out"}, slots)
            if step == 0:
                self.assertEqual([r["env_id"] for r in records], [1])
            elif step in (1, 3):
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["env_id"], 0)
                self.assertEqual(records[0]["length"], 2)
                self.assertEqual(records[0]["return"], 4.)
            else:
                self.assertEqual(records, [])
        summary = stats.summary()
        self.assertEqual(summary["completed_episodes"], 3)
        self.assertEqual(summary["mean_episode_return"], 11 / 3)
        self.assertEqual(summary["mean_episode_length"], 5 / 3)
        self.assertEqual(summary["discarded_partial_episodes"], 0)

    def test_partial_attempts_do_not_enter_success_rate(self):
        stats = EpisodeStatistics(2, 2)
        self.assertIsNone(stats.summary()["success_rate"])
        stats.update(torch.ones(2), torch.tensor([True, False]),
                     {"success": torch.tensor([True, False])}, set(), torch.tensor([0, 1]))
        summary = stats.summary()
        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(summary["success_rate"], 1.)
        self.assertEqual(summary["discarded_partial_episodes"], 1)
        self.assertFalse(summary["complete"])


if __name__ == "__main__":
    unittest.main()
