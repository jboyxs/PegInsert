"""Batched geometric expert producing the SAME raw joint-delta actions as PPO.

The expert reads privileged geometry, but labels the unmodified policy observations.
It never writes robot/object state or joint targets during the rollout. Quaternions
use Isaac Lab 3's xyzw convention throughout.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
import isaaclab.utils.math as math_utils

from .config.franka.task_cfg import ASSEMBLY_PROFILE, ASSEMBLY_TIP, HELD_ASSET_ALIGN_OFFSET


class Stage(IntEnum):
    GRASP = 0
    LIFT = 1
    TRANSFER = 2
    APPROACH = 3
    INSERT = 4
    RETRACT = 5
    FAILED = 6


@dataclass(frozen=True)
class ExpertCfg:
    settle_s: float = 0.32
    clearance: float = 0.08
    approach_gap: float = 0.003
    position_step: float = 0.01
    rotation_step: float = 0.12
    insertion_step: float = 0.00035
    xy_tolerance: float = 0.00015
    rotation_tolerance: float = 0.006
    stall_s: float = 2.0
    stall_progress: float = 0.0003
    max_retries: int = 3
    damping: float = 0.015
    rotation_weight: float = 0.15
    max_action: float = 2.0
    max_joint_step: float = 0.05
    joint_limit_margin: float = 0.005
    max_grasp_slip: float = 0.025
    velocity_damping_s: float = 0.005
    integral_gain: float = 0.25


def limit_norm(value, limit):
    return value * (limit / value.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)


def nearest_square_orientation(peg_quat, hole_quat):
    """Choose a fixed symmetry branch for the episode, avoiding yaw branch chatter."""
    yaw = torch.arange(4, device=peg_quat.device, dtype=peg_quat.dtype) * (torch.pi / 2)
    zeros = torch.zeros_like(yaw)
    symmetry = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    candidates = math_utils.quat_mul(
        hole_quat[:, None].expand(-1, 4, -1).reshape(-1, 4),
        symmetry[None].expand(len(peg_quat), -1, -1).reshape(-1, 4),
    ).reshape(-1, 4, 4)
    index = (candidates * peg_quat[:, None]).sum(-1).abs().argmax(-1)
    return candidates[torch.arange(len(peg_quat), device=peg_quat.device), index]


def joint_delta_to_action(delta, joint_pos, limits, scale, cfg):
    """Invert RelativeJointPositionAction's scale, with direction-preserving bounds.

    Scale describes the *target offset*, not a velocity or a per-episode absolute
    target. Decimation remains entirely under the environment's action manager.
    """
    capacity = torch.minimum(scale.abs() * cfg.max_action, torch.full_like(scale, cfg.max_joint_step))
    ratio = (delta.abs() / capacity).amax(-1, keepdim=True).clamp_min(1.0)
    delta = delta / ratio
    lower, upper = limits[..., 0] + cfg.joint_limit_margin, limits[..., 1] - cfg.joint_limit_margin
    # Already-at-limit joints must still be allowed to move back into the range.
    delta = torch.maximum(torch.minimum(delta, (upper - joint_pos).clamp_min(0)),
                          (lower - joint_pos).clamp_max(0))
    return delta / scale


class PegInsertExpert:
    def __init__(self, env, cfg=None):
        self.env = env
        self.cfg = cfg or ExpertCfg()
        self.robot = env.scene["robot"]
        manager = env.action_manager
        if manager.active_terms != ["arm_action", "gripper_action"] or manager.action_term_dim != [7, 1]:
            raise ValueError("Expert requires the actor's [7 relative arm joints, 1 binary gripper] action layout")
        arm = manager.get_term("arm_action")
        from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
        from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
        if not isinstance(arm, RelativeJointPositionAction) or not isinstance(manager.get_term("gripper_action"), BinaryJointPositionAction):
            raise ValueError("Expert requires relative arm position actions and binary gripper position actions")
        if not arm.cfg.use_zero_offset or arm.cfg.clip is not None:
            raise ValueError("Expert expects zero-offset, unclipped actor action terms")
        # Read the runtime action term: this is the resolved actor ordering and scale.
        self.joint_ids = arm._joint_ids
        self.joint_names = list(arm._joint_names)
        self.scale = torch.as_tensor(arm._scale, device=env.device).expand(env.num_envs, 7).clone()
        if not torch.isfinite(self.scale).all() or (self.scale == 0).any():
            raise ValueError("Arm action scales must be finite and nonzero")
        body_ids, _ = self.robot.find_bodies("panda_fingertip_centered")
        self.body_id = body_ids[0]
        self.jacobian_id = self.body_id - int(self.robot.is_fixed_base)
        self.jacobian_joint_ids = self.joint_ids + self.robot.num_base_dofs
        self.stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.age = torch.zeros_like(self.stage)
        self.retries = torch.zeros_like(self.stage)
        self.stall = torch.zeros_like(self.stage)
        self.best_z = torch.full((env.num_envs,), float("inf"), device=env.device)
        self.lift_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.lift_z = torch.zeros(env.num_envs, device=env.device)
        self.lift_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self.lift_quat[:, 3] = 1
        self.target_quat = torch.zeros_like(self.lift_quat)
        self.target_quat[:, 3] = 1
        self.grasp_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self.failure_reason = torch.zeros_like(self.stage)  # 1: slip; 2: retries exhausted
        self.position_integral = torch.zeros(env.num_envs, 3, device=env.device)
        self.rotation_integral = torch.zeros_like(self.position_integral)

    def reset(self, ids):
        self.stage[ids] = Stage.GRASP
        self.age[ids] = 0
        self.retries[ids] = 0
        self.stall[ids] = 0
        self.best_z[ids] = float("inf")
        self.failure_reason[ids] = 0
        self.position_integral[ids] = 0
        self.rotation_integral[ids] = 0

    def _enter(self, mask, stage):
        self.stage[mask] = stage
        self.age[mask] = 0
        self.position_integral[mask] = 0
        self.rotation_integral[mask] = 0

    @torch.no_grad()
    def act(self):
        cfg = self.cfg
        hole_p, hole_q = ASSEMBLY_TIP.apply(self.env.scene["fixed_asset"])
        tip_p, tip_q = HELD_ASSET_ALIGN_OFFSET.apply(self.env.scene["held_asset"])
        ee_p = self.robot.data.body_link_pos_w.torch[:, self.body_id]
        ee_q = self.robot.data.body_link_quat_w.torch[:, self.body_id]
        tip_local, _ = math_utils.subtract_frame_transforms(hole_p, hole_q, tip_p, tip_q)
        grasp_p, grasp_q = math_utils.subtract_frame_transforms(ee_p, ee_q, tip_p, tip_q)
        self.age += 1
        # The reset places fingers around the asset with a randomized opening.
        # Close physically and wait before measuring the actual grasp transform.
        ready = (self.stage == Stage.GRASP) & (self.age * self.env.step_dt >= cfg.settle_s)
        self.lift_xy[ready] = tip_local[ready, :2]
        self.lift_z[ready] = torch.maximum(tip_local[ready, 2] + 0.04,
                                         torch.full_like(tip_local[ready, 2], cfg.clearance))
        self.lift_quat[ready] = tip_q[ready]
        self.target_quat[ready] = nearest_square_orientation(tip_q[ready], hole_q[ready])
        self.grasp_pos[ready] = grasp_p[ready]
        self._enter(ready, Stage.LIFT)
        slip = (grasp_p - self.grasp_pos).norm(dim=-1)
        lost = (self.stage != Stage.GRASP) & (slip > cfg.max_grasp_slip)
        self.failure_reason[lost] = 1
        self._enter(lost, Stage.FAILED)

        rotation_error = math_utils.axis_angle_from_quat(math_utils.quat_mul(self.target_quat, math_utils.quat_inv(tip_q)))
        xy = tip_local[:, :2].norm(dim=-1)
        aligned = (xy < cfg.xy_tolerance) & (rotation_error.norm(dim=-1) < cfg.rotation_tolerance)
        # Use a snapshot so each env advances by at most one stage per control step.
        old = self.stage.clone()
        self._enter((old == Stage.LIFT) & (tip_local[:, 2] >= self.lift_z - 0.003), Stage.TRANSFER)
        coarse_aligned = (xy < 0.002) & (rotation_error.norm(dim=-1) < 0.02)
        self._enter((old == Stage.TRANSFER) & coarse_aligned, Stage.APPROACH)
        mouth = cfg.approach_gap
        self._enter((old == Stage.APPROACH) & aligned & (tip_local[:, 2] <= mouth + 0.0008), Stage.INSERT)
        inserting = self.stage == Stage.INSERT
        improved = tip_local[:, 2] < self.best_z - cfg.stall_progress
        self.best_z = torch.where(inserting & improved, tip_local[:, 2], self.best_z)
        self.stall = torch.where(inserting & ~improved, self.stall + 1, torch.zeros_like(self.stall))
        blocked = inserting & ((self.stall * self.env.step_dt > cfg.stall_s) | (xy > 0.002))
        self.retries[blocked] += 1
        self.lift_z[blocked] = 0.02
        self._enter(blocked, Stage.RETRACT)
        exhausted = blocked & (self.retries > cfg.max_retries)
        self.failure_reason[exhausted] = 2
        self._enter(exhausted, Stage.FAILED)
        self._enter((old == Stage.RETRACT) & (tip_local[:, 2] >= 0.012), Stage.TRANSFER)
        self.best_z[self.stage != Stage.INSERT] = float("inf")

        desired_local = torch.zeros_like(tip_local)
        desired_local[:, 2] = self.lift_z
        lift = self.stage == Stage.LIFT
        desired_local[lift, :2] = self.lift_xy[lift]
        desired_local[self.stage == Stage.APPROACH, 2] = mouth
        # Advance from measured depth; never accumulate an unreachable downward target.
        # Keep the final target just above the seated plane to avoid driving through it.
        seated_z = ASSEMBLY_PROFILE.segments[0].start_pose.pos[2] - ASSEMBLY_TIP.pos[2] + 0.0004
        desired_local[inserting, 2] = (tip_local[inserting, 2] - cfg.insertion_step).clamp_min(seated_z)
        # Correct lateral/rotational error before increasing insertion pressure.
        desired_local[inserting & ~aligned, 2] = tip_local[inserting & ~aligned, 2]
        retract = self.stage == Stage.RETRACT
        desired_local[retract, :2] = tip_local[retract, :2]
        desired_local[retract, 2] = 0.015
        desired_tip_q = torch.where(lift[:, None], self.lift_quat, self.target_quat)
        desired_tip_p, _ = math_utils.combine_frame_transforms(hole_p, hole_q, desired_local)
        # Solve T_ee_goal = T_tip_goal * inv(T_ee_tip), accounting for off-center grasps.
        inv_grasp_q = math_utils.quat_inv(grasp_q)
        inv_grasp_p = -math_utils.quat_apply(inv_grasp_q, grasp_p)
        goal_p, goal_q = math_utils.combine_frame_transforms(desired_tip_p, desired_tip_q, inv_grasp_p, inv_grasp_q)
        near_contact = (self.stage == Stage.APPROACH) | (self.stage == Stage.INSERT)
        step_limit = torch.where(near_contact[:, None], 0.003, cfg.position_step)
        raw_position_error = goal_p - ee_p
        raw_angle_error = math_utils.axis_angle_from_quat(math_utils.quat_mul(goal_q, math_utils.quat_inv(ee_q)))
        # Integrate only close to a waypoint: compensate friction/load without
        # winding up during the long transfer. This is still a raw delta action.
        integrate = ((raw_position_error.norm(dim=-1) < 0.003)
                     & (raw_angle_error.norm(dim=-1) < 0.03)
                     & (self.stage != Stage.GRASP) & (self.stage != Stage.FAILED))
        self.position_integral = torch.where(integrate[:, None],
            limit_norm(self.position_integral + self.env.step_dt * raw_position_error, 0.003),
            torch.zeros_like(self.position_integral))
        self.rotation_integral = torch.where(integrate[:, None],
            limit_norm(self.rotation_integral + self.env.step_dt * raw_angle_error, 0.02),
            torch.zeros_like(self.rotation_integral))
        position_error = limit_norm(0.7 * raw_position_error + cfg.integral_gain * self.position_integral, step_limit)
        angle_error = limit_norm(0.7 * raw_angle_error + cfg.integral_gain * self.rotation_integral, cfg.rotation_step)
        error = torch.cat((position_error, angle_error * cfg.rotation_weight), dim=-1)
        jacobian_w = self.robot.data.body_link_jacobian_w.torch[:, self.jacobian_id, :, self.jacobian_joint_ids]
        jacobian = jacobian_w.clone()
        jacobian[:, 3:] *= cfg.rotation_weight
        eye = torch.eye(6, device=self.env.device)
        delta = (jacobian.mT @ torch.linalg.solve(jacobian @ jacobian.mT + cfg.damping**2 * eye,
                                                error.unsqueeze(-1))).squeeze(-1)
        # The robot has gravity compensation, the grasped peg does not. Offset
        # the implicit joint PD drives by J^T*wrench / stiffness to support its
        # known weight, avoiding a persistent millimeter-scale tracking error.
        held = self.env.scene["held_asset"]
        gravity = torch.tensor(self.env.cfg.sim.gravity, device=self.env.device)
        force = -held.data.body_mass.torch * gravity
        lever = held.data.body_com_pos_w.torch[:, 0] - ee_p
        wrench = torch.cat((force, torch.linalg.cross(lever, force)), dim=-1)
        stiffness = self.robot.data.joint_stiffness.torch[:, self.joint_ids]
        delta += (jacobian_w.mT @ wrench.unsqueeze(-1)).squeeze(-1) / stiffness.clamp_min(1.0)
        joint_pos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        # Relative targets behave as torque-producing offsets. Add velocity
        # feedback to brake at waypoints rather than oscillating around them.
        delta -= cfg.velocity_damping_s * self.robot.data.joint_vel.torch[:, self.joint_ids]
        limits = self.robot.data.joint_pos_limits.torch[:, self.joint_ids]
        action = torch.zeros(self.env.num_envs, 8, device=self.env.device)
        action[:, :7] = joint_delta_to_action(delta, joint_pos, limits, self.scale, cfg)
        action[(self.stage == Stage.GRASP) | (self.stage == Stage.FAILED), :7] = 0
        action[:, 7] = -1.0  # BinaryJointPositionAction: negative means CLOSED.
        if not torch.isfinite(action).all():
            raise FloatingPointError("Non-finite expert action; stopping collection")
        diagnostics = torch.stack((xy, tip_local[:, 2], rotation_error.norm(dim=-1), slip), dim=-1)
        return action, {"stage": self.stage.clone(), "geometry": diagnostics}

    def metadata(self):
        return {
            "joint_names": self.joint_names, "arm_scale": self.scale[0].cpu().tolist(),
            "action_layout": "[7 raw relative joint position commands, binary gripper (-1=close)]",
            "geometry_columns": ["xy_error_m", "tip_height_above_mouth_m", "rotation_error_rad", "grasp_slip_m"],
            "stages": {stage.name: int(stage) for stage in Stage},
        }
