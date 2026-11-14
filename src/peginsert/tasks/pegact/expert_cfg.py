"""Compose the expert run without changing PPO's training configuration."""

from isaaclab.app import validate_launch
from isaaclab.managers import ManagerTermBaseCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from isaaclab_tasks.utils.hydra import resolve_task_config


def air_grasp_ready(env, env_ids):
    """Reject IK misses and wide-open fingers before gravity can drop the peg.

    This is an additional acceptance condition alongside the collision checks;
    the expert configuration also narrows the in-hand perturbations below.
    """
    import isaaclab.utils.math as math_utils
    from .config.franka.task_cfg import HELD_ASSET_GRASP_POINT
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("panda_fingertip_centered")
    finger_ids, _ = robot.find_joints("panda_finger_.*")
    grasp_p, _ = HELD_ASSET_GRASP_POINT.apply(env.scene["held_asset"])
    local, _ = math_utils.subtract_frame_transforms(
        robot.data.body_link_pos_w.torch[:, body_ids[0]],
        robot.data.body_link_quat_w.torch[:, body_ids[0]], grasp_p,
    )
    opening = robot.data.joint_pos.torch[:, finger_ids].amax(dim=-1)
    return ((local[:, :2].norm(dim=-1) < 0.0015) & (local[:, 2].abs() < 0.006)
            & (opening < 0.006))[env_ids]


def make_expert_env_cfg(*, num_envs=16, device="cuda:0", seed=42, state_table_size=512, headless=False):
    if num_envs < 1 or state_table_size < 1:
        raise ValueError("num_envs and state_table_size must be positive")
    cfg, _ = resolve_task_config("PegInset", None, play_mode=True, overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    # play_mode selects only grasp_asset_in_air and disables the gravity curriculum.
    cfg.curriculum.difficulty_scheduler = None
    cfg.events.reset_strategies.params["state_table_size"] = state_table_size
    cfg.events.reset_strategies.params["acceptance_conditions"]["air_grasp_ready"] = ManagerTermBaseCfg(func=air_grasp_ready)
    air = cfg.events.reset_strategies.params["reset_term"].params["terms"]["reset_strategies"].params["terms"]["grasp_asset_in_air"]
    grasp_ik = air.params["terms"]["reset_end_effector_around_held_asset"].params
    # Start with a physical grasp of the randomly tilted peg. The training reset
    # aims the hand down in WORLD axes, which can miss a tilted air asset.
    grasp_ik["orientation_in_asset_frame"] = True
    grasp_ik["ik_iterations"] = (40, 60)
    grasp_ik["pose_range_b"] = {
        "x": (-0.0005, 0.0005), "y": (-0.0005, 0.0005), "z": (-0.003, 0.003),
        "roll": (3.14159 - 0.02, 3.14159 + 0.02), "pitch": (-0.02, 0.02), "yaw": (-3.14, 3.14),
    }
    cfg.sim.visualizer_cfgs = [] if headless else [
        NewtonGLVisualizerCfg(
            eye=(0.9, 0.8, 0.6), lookat=(0.45, 0.0, 0.08),
            max_visible_envs=4, randomly_sample_visible_envs=False,
            streaming_view=True, streaming_envs=[0],
            streaming_cam_target_prim_path=cfg.scene.fixed_asset.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/*"),
            streaming_cam_eye=(0.25, 0.3, 0.25),
            streaming_cam_renderer_cfg=NewtonWarpRendererCfg(),
        )
    ]
    cfg.validate()
    _, kit_sources = validate_launch(cfg, {"headless": headless, "livestream": 0})
    if kit_sources:
        raise RuntimeError(f"Expert collection must be Kit-free; found {kit_sources}")
    return cfg
