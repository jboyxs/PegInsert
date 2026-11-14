"""Verify the actor action contract, geometric control, and BC episode boundaries."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import isaaclab.utils.math as math_utils
from isaaclab.app import validate_launch
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

from peginsert.tasks.pegact.demonstrations import DemonstrationWriter
from peginsert.tasks.pegact.expert import (
    ExpertCfg, PegInsertExpert, Stage, joint_delta_to_action, nearest_square_orientation,
)
from peginsert.tasks.pegact.expert_cfg import make_expert_env_cfg
from peginsert.tasks.pegact.mdp.events import reset_end_effector_around_asset


def proxy(value):
    return SimpleNamespace(torch=value)


def fake_env():
    """Two rigid grasps with a known world-frame Jacobian, without simulation."""
    identity = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    tip_p = torch.tensor([[0.03, 0.01, 0.05], [0.0, 0.0, 0.025]])
    ee_p = tip_p + torch.tensor([0., 0., 0.035])
    jacobian = torch.zeros(2, 1, 6, 9)
    jacobian[:, 0, :, :6] = torch.eye(6)
    robot = SimpleNamespace(
        is_fixed_base=True, num_base_dofs=0,
        find_bodies=lambda name: ([1], [name]),
        data=SimpleNamespace(
            body_link_pos_w=proxy(torch.stack((ee_p, ee_p), dim=1)),
            body_link_quat_w=proxy(torch.stack((identity, identity), dim=1)),
            joint_pos=proxy(torch.zeros(2, 9)),
            joint_vel=proxy(torch.zeros(2, 9)),
            joint_stiffness=proxy(torch.full((2, 9), 80.)),
            joint_pos_limits=proxy(torch.tensor([-2., 2.]).expand(2, 9, 2).clone()),
            body_link_jacobian_w=proxy(jacobian),
        ),
    )
    arm = object.__new__(RelativeJointPositionAction)
    arm.cfg = SimpleNamespace(use_zero_offset=True, clip=None)
    arm._joint_ids = torch.arange(7)
    arm._joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    arm._scale = torch.tensor([[0.02] * 6 + [0.2]]).expand(2, 7)
    gripper = object.__new__(BinaryJointPositionAction)
    terms = {"arm_action": arm, "gripper_action": gripper}
    return SimpleNamespace(
        device="cpu", num_envs=2, step_dt=0.04,
        cfg=SimpleNamespace(sim=SimpleNamespace(gravity=(0., 0., -9.81))),
        action_manager=SimpleNamespace(active_terms=list(terms), action_term_dim=[7, 1], get_term=terms.__getitem__),
        scene={
            "robot": robot,
            "held_asset": SimpleNamespace(data=SimpleNamespace(root_pos_w=proxy(tip_p), root_quat_w=proxy(identity.clone()),
                body_mass=proxy(torch.full((2, 1), 0.019)), body_com_pos_w=proxy(tip_p[:, None] + torch.tensor([0., 0., 0.025])))),
            "fixed_asset": SimpleNamespace(data=SimpleNamespace(root_pos_w=proxy(torch.zeros(2, 3)), root_quat_w=proxy(identity.clone()))),
        },
    )


class ExpertTests(unittest.TestCase):
    def test_relative_grasp_orientation_is_opt_in(self):
        env = fake_env()
        reset = object.__new__(reset_end_effector_around_asset)
        reset._env = env
        reset.robot = env.scene["robot"]
        reset.ranges = torch.zeros(6, 2)
        reset.joint_ids = torch.arange(7)
        reset.fixed_asset = env.scene["held_asset"]
        yaw = torch.tensor([0.7, -1.2])
        q = math_utils.quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), yaw)
        reset.fixed_asset_offset = SimpleNamespace(apply=lambda asset: (torch.zeros(2, 3), q))
        reset.robot.data.root_link_pos_w = proxy(torch.zeros(2, 3))
        reset.robot.data.root_link_quat_w = proxy(torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
        commands = []
        reset.solver = SimpleNamespace(
            _compute_frame_pose=lambda: (torch.zeros(2, 3), torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)),
            process_actions=lambda command: commands.append(command.clone()),
        )
        for relative in (False, True):
            reset(env, torch.tensor([0, 1]), None, None, {}, None,
                  ik_iterations=(0, 0), orientation_in_asset_frame=relative)
        torch.testing.assert_close(commands[0][:, 3:], torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
        torch.testing.assert_close(commands[1][:, 3:], q)

    def test_newton_warp_configuration_is_kit_free_and_air_only(self):
        for headless in (False, True):
            cfg = make_expert_env_cfg(num_envs=2, state_table_size=4, headless=headless)
            _, kit = validate_launch(cfg, {"livestream": 0, "headless": headless})
            self.assertFalse(kit)
            self.assertIsNone(cfg.events.variable_gravity)
            self.assertIsNone(cfg.curriculum.difficulty_scheduler)
            self.assertEqual(cfg.sim.gravity, (0., 0., -9.81))
            choices = cfg.events.reset_strategies.params["reset_term"].params["terms"]["reset_strategies"].params["terms"]
            self.assertEqual(list(choices), ["grasp_asset_in_air"])
            self.assertEqual(cfg.actions.arm_action.scale["panda_joint7"], 0.2)
            if not headless:
                self.assertEqual(cfg.sim.visualizer_cfgs[0].streaming_cam_renderer_cfg.renderer_type, "newton_warp")

    def test_raw_action_reconstructs_actor_target_and_respects_joint_limits(self):
        cfg = ExpertCfg()
        scale = torch.tensor([[0.02] * 6 + [0.2]])
        q = torch.zeros(1, 7)
        delta = torch.tensor([[0.01] * 6 + [0.02]])
        limits = torch.tensor([-1., 1.]).expand(1, 7, 2)
        raw = joint_delta_to_action(delta, q, limits, scale, cfg)
        torch.testing.assert_close(raw, torch.tensor([[0.5] * 6 + [0.1]]))
        # Exercise the actual upstream action implementation, not just the inverse.
        arm = object.__new__(RelativeJointPositionAction)
        arm._processed_actions = raw * scale
        arm._joint_ids = torch.arange(7)
        targets = []
        arm._asset = SimpleNamespace(data=SimpleNamespace(joint_pos=proxy(q)),
                                     set_joint_position_target_index=lambda **kw: targets.append(kw["target"].clone()))
        arm.apply_actions()
        torch.testing.assert_close(targets[0], q + delta)
        q[:, 0] = 0.994
        raw = joint_delta_to_action(torch.ones_like(q), q, limits, scale, cfg)
        self.assertLessEqual(float((q + raw * scale)[0, 0]), 0.995001)
        self.assertLessEqual(float(raw.abs().max()), cfg.max_action)
        self.assertLessEqual(float((raw * scale).abs().max()), cfg.max_joint_step + 1e-7)

    def test_square_symmetry_and_quaternion_sign_invariance(self):
        yaw = torch.tensor([1.5, -1.5, 3.1])
        z = torch.zeros(3)
        peg = math_utils.quat_from_euler_xyz(z, z, yaw)
        hole = torch.tensor([[0., 0., 0., 1.]]).expand(3, 4)
        chosen = nearest_square_orientation(peg, hole)
        torch.testing.assert_close(chosen, nearest_square_orientation(-peg, hole))
        difference = math_utils.axis_angle_from_quat(math_utils.quat_mul(chosen, math_utils.quat_inv(peg)))
        self.assertTrue((difference.norm(dim=-1) < 0.1).all())

    def test_rigid_grasp_lift_moves_tip_up_and_partial_reset_preserves_other_env(self):
        env = fake_env()
        expert = PegInsertExpert(env, ExpertCfg(settle_s=0.04))
        action, diagnostics = expert.act()
        self.assertTrue((expert.stage == Stage.LIFT).all())
        self.assertTrue((action[:, 2] > 0).all())
        self.assertTrue((action[:, 7] == -1).all())
        self.assertTrue(torch.isfinite(diagnostics["geometry"]).all())
        expert.reset(torch.tensor([0]))
        self.assertEqual(int(expert.stage[0]), Stage.GRASP)
        self.assertEqual(int(expert.stage[1]), Stage.LIFT)
        self.assertEqual(int(expert.age[0]), 0)

    def test_stall_retracts_and_retry_exhaustion_is_explicit(self):
        env = fake_env()
        expert = PegInsertExpert(env, ExpertCfg(settle_s=0.04, stall_s=0.04, max_retries=0))
        expert.act()  # Establish grasp reference.
        env.scene["held_asset"].data.root_pos_w.torch[:, :2] = 0
        env.scene["robot"].data.body_link_pos_w.torch[:, 1, :2] = 0
        expert.stage[:] = Stage.INSERT
        for _ in range(3):
            expert.act()
        self.assertTrue((expert.stage == Stage.FAILED).all())
        self.assertTrue((expert.failure_reason == 2).all())


class DemonstrationTests(unittest.TestCase):
    def test_async_episodes_keep_pre_action_history_and_never_cross_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DemonstrationWriter(directory, 2, {"seed": 42}, validation_fraction=0)
            for step in range(3):
                observation = torch.full((2, 15), float(step))
                action = torch.full((2, 8), float(step + 10))
                terminal = torch.tensor([step in (0, 2), step == 2])
                success = terminal.clone()
                diagnostics = {"stage": torch.zeros(2, dtype=torch.long), "failure": torch.tensor([False, step == 2]),
                               "expert_failure": torch.zeros(2, dtype=torch.long)}
                writer.append(observation, action, torch.ones(2), terminal, torch.zeros(2, dtype=torch.bool), success, diagnostics)
                observation.fill_(-999)  # Simulate env reusing/overwriting its observation buffer.
                action.fill_(-999)
            summary = writer.close()
            self.assertEqual(summary["completed_episodes"], 3)
            self.assertEqual(summary["saved_episodes"], 2)
            first = torch.load(Path(directory) / "train/episode_000000.pt", weights_only=True)
            second = torch.load(Path(directory) / "train/episode_000001.pt", weights_only=True)
            torch.testing.assert_close(first["policy_observation"], torch.zeros(1, 15))
            torch.testing.assert_close(first["action"], torch.full((1, 8), 10.))
            torch.testing.assert_close(second["policy_observation"][:, 0], torch.tensor([1., 2.]))
            records = [json.loads(line) for line in (Path(directory) / "episodes.jsonl").read_text().splitlines()]
            self.assertFalse(records[-1]["success"])  # Success + abnormal failure is not a demonstration.
            self.assertIsNone(records[-1]["file"])

    def test_partial_and_nonfinite_episodes_are_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DemonstrationWriter(directory, 1, {})
            values = (torch.ones(1, 5), torch.ones(1, 8), torch.ones(1), torch.tensor([False]),
                      torch.tensor([False]), torch.tensor([False]), {"failure": torch.tensor([False])})
            writer.append(*values)
            values[0][0, 0] = float("nan")
            with self.assertRaises(FloatingPointError):
                writer.append(*values)
            summary = writer.close(interrupted=True)
            self.assertEqual(summary["discarded_partial_episodes"], 1)
            self.assertFalse(list(Path(directory).rglob("*.pt")))
            with self.assertRaises(FileExistsError):
                DemonstrationWriter(directory, 1, {})


if __name__ == "__main__":
    unittest.main()
