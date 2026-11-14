"""Evaluate an RSL-RL checkpoint on original air-grasp resets at normal gravity.

Example:
    uv run python scripts/evaluate.py --checkpoint logs/rsl_rl/peg_inset/RUN/model_30000.pt --headless --episodes 1000
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local RSL-RL training checkpoint (.pt)")
    parser.add_argument("--agent_cfg", type=Path, help="Defaults to params/agent.yaml beside the checkpoint, if present")
    parser.add_argument("--episodes", type=int, default=1000, help="Exact number of complete attempts")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--state_table_size", type=int, default=512, help="Number of independently generated air-grasp states")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation/reset-bank seed; use a seed different from training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physics", choices=["newton_mjwarp", "isaacsim_physx", "physx"], default="newton_mjwarp",
                        help="Use the same backend as training (default: newton_mjwarp)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, help="Optional vector-step limit; unfinished runs are marked incomplete")
    parser.add_argument("--output", type=Path, help="Empty result directory (default: logs/evaluation/<timestamp>)")
    parser.add_argument("--success_gifs", type=int, default=0, help="Save the first N successful episodes as GIFs (Newton, <=4 envs)")
    args = parser.parse_args()
    if min(args.episodes, args.num_envs, args.state_table_size) < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes, num_envs, state_table_size, and max_steps must be positive")
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.agent_cfg is not None:
        args.agent_cfg = args.agent_cfg.resolve()
        if not args.agent_cfg.is_file():
            parser.error(f"Agent configuration does not exist: {args.agent_cfg}")
    else:
        candidate = args.checkpoint.parent / "params" / "agent.yaml"
        args.agent_cfg = candidate if candidate.is_file() else None
    args.num_envs = min(args.num_envs, args.episodes)
    if args.success_gifs < 0 or args.success_gifs > args.episodes:
        parser.error("success_gifs must be between 0 and episodes")
    if args.success_gifs and (args.physics != "newton_mjwarp" or args.num_envs > 4):
        parser.error("GIF capture requires --physics newton_mjwarp and --num_envs <= 4")
    return args


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    output = (args.output or Path("logs/evaluation") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty result directory: {output}")

    import peginsert  # noqa: F401 -- Register the local task.
    from isaaclab.app import launch_simulation
    from isaaclab.utils import update_from_dict
    from isaaclab.utils.io import dump_yaml, load_yaml
    from peginsert.tasks.pegact.evaluation import EpisodeStatistics, make_evaluation_env_cfg

    cfg, agent_cfg = make_evaluation_env_cfg(
        num_envs=args.num_envs, device=args.device, seed=args.seed,
        state_table_size=args.state_table_size, physics=args.physics, headless=args.headless,
    )
    if args.success_gifs:
        # Register the renderer before cloning so Newton includes visual geometry.
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        cfg.scene.evaluation_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/EvaluationCamera", width=640, height=480,
            update_period=0.0, data_types=["rgb"],
            spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                   clipping_range=(0.01, 100.0)),
            renderer_cfg=NewtonWarpRendererCfg(use_cuda_graph=False),
        )
    if args.agent_cfg:
        update_from_dict(agent_cfg, load_yaml(str(args.agent_cfg)))
    else:
        print("[eval] No params/agent.yaml found; using the project's default network configuration.", flush=True)
    agent_cfg.device = args.device
    agent_cfg.seed = args.seed
    cfg.log_dir = str(output)
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(output / "env.yaml"), cfg)
    dump_yaml(str(output / "agent.yaml"), agent_cfg)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "task": "PegInset", "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "agent_cfg_source": str(args.agent_cfg) if args.agent_cfg else "project_default",
        "physics": args.physics, "gravity": list(cfg.sim.gravity), "reset_strategy": "grasp_asset_in_air",
        "initial_state_sampling": "uniform_with_replacement", "state_table_size": args.state_table_size,
        "seed": args.seed, "num_envs": args.num_envs, "device": args.device,
        "policy": "deterministic_mean", "success_rule": "success termination without any failure termination",
        "episode_assignment": "fixed per-env quotas", "max_steps": args.max_steps,
        "requested_success_gifs": args.success_gifs,
    }
    write_json(output / "metadata.json", metadata)
    with launch_simulation(cfg, {"headless": args.headless, "livestream": 0}):
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
        )

        env = gym.make("PegInset", cfg=cfg).unwrapped
        stats = None
        gif_recorder = None
        steps = 0
        started = time.monotonic()
        stop_reason = "error"
        try:
            with torch.inference_mode():
                # The wrapper resets once; the cfg seed is applied before bank generation.
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
                runner = create_rsl_rl_runner(wrapped, agent_cfg)
                runner.load(str(args.checkpoint), load_cfg={"actor": True, "iteration": True}, map_location=args.device)
                policy = runner.get_inference_policy(device=env.device)
                metadata["checkpoint_iteration"] = runner.current_learning_iteration
                metadata["policy_observation_shape"] = list(env.obs_buf["policy"].shape[1:])
                write_json(output / "metadata.json", metadata)
                stats = EpisodeStatistics(env.num_envs, args.episodes, env.device)
                manager = env.termination_manager
                timeout_terms = {name for name in manager.active_terms if manager.get_term_cfg(name).time_out}
                bank = env.event_manager.get_term_cfg("reset_strategies").func
                obs = wrapped.get_observations()
                if args.success_gifs:
                    from peginsert.tasks.pegact.evaluation_gifs import SuccessGifRecorder

                    gif_recorder = SuccessGifRecorder(env, output, args.success_gifs)
                print(f"[eval] checkpoint={args.checkpoint}\n[eval] results={output}", flush=True)
                with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                    while stats.completed < args.episodes:
                        if args.max_steps is not None and steps >= args.max_steps:
                            stop_reason = "max_steps"
                            break
                        if not env.sim.is_running():
                            stop_reason = "simulation_stopped"
                            break
                        # Auto-reset replaces the sampled slot during step(), so capture it first.
                        slots = bank.sampled_slots.clone()
                        action = policy(obs)
                        if not torch.isfinite(action).all():
                            raise FloatingPointError("Policy produced a non-finite action")
                        obs, rewards, dones, _ = wrapped.step(action)
                        if not torch.isfinite(rewards).all() or not torch.isfinite(obs["policy"]).all():
                            raise FloatingPointError("Environment produced non-finite rewards or observations")
                        flags = {name: manager.get_term(name) for name in manager.active_terms}
                        records = stats.update(rewards, dones, flags, timeout_terms, slots)
                        if gif_recorder is not None:
                            gif_recorder.after_step(dones, records)
                        policy.reset(dones)
                        steps += 1
                        for record in records:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                        if records:
                            stream.flush()
                            print(f"[eval] episodes={stats.completed}/{args.episodes} success={stats.successes} "
                                  f"failure={stats.failures} timeout={stats.timeouts} "
                                  f"success_rate={stats.successes / stats.completed:.2%}", flush=True)
                    else:
                        stop_reason = "completed"
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            print("[eval] Interrupted; saving complete episodes only.", flush=True)
        finally:
            if stats is not None:
                summary = {
                    **metadata, **stats.summary(), "stop_reason": stop_reason,
                    "vector_steps": steps, "elapsed_s": round(time.monotonic() - started, 2),
                }
                if gif_recorder is not None:
                    summary["success_gifs"] = gif_recorder.saved
                    summary["gifs_complete"] = len(gif_recorder.saved) == args.success_gifs
                write_json(output / "summary.json", summary)
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            if gif_recorder is not None:
                gif_recorder.close()
            env.close()


if __name__ == "__main__":
    main()
