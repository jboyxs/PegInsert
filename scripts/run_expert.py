"""Run the Newton/Warp expert and collect episode files for behavior cloning.

Examples:
    uv run python scripts/run_expert.py --num_envs 16 --episodes 100 --headless
    uv run python scripts/run_expert.py --num_envs 1 --episodes 5
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=100, help="Number of completed attempts (successes + failures)")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional vector-step limit for smoke tests")
    parser.add_argument("--state_table_size", type=int, default=512, help="Collision-validated grasp_asset_in_air initial states")
    parser.add_argument("--reset_strategy", choices=["grasp_asset_in_air"], default="grasp_asset_in_air")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Disable the Newton viewer and image rendering")
    parser.add_argument("--output", type=Path, default=None, help="Empty dataset directory")
    parser.add_argument("--keep_failures", action="store_true", help="Also save failed episodes; filter episode_success for BC")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--insertion_step", type=float, default=0.00035, help="Maximum insertion target advance [m/control step]")
    args = parser.parse_args()
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1) or args.insertion_step <= 0:
        parser.error("episodes, max_steps and insertion_step must be positive")
    if args.num_envs < 1 or args.state_table_size < 1 or not 0 <= args.validation_fraction < 1:
        parser.error("num_envs/state_table_size must be positive and validation_fraction in [0, 1)")
    return args


def main():
    args = parse_args()
    # No AppLauncher: that deprecated adapter unconditionally starts Kit.
    import peginsert  # noqa: F401 -- Gym registration
    from isaaclab.app import launch_simulation
    from peginsert.tasks.pegact.expert_cfg import make_expert_env_cfg

    cfg = make_expert_env_cfg(num_envs=args.num_envs, device=args.device, seed=args.seed,
                              state_table_size=args.state_table_size, headless=args.headless)
    output = args.output or Path("logs/expert") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Fail before costly simulator startup if output would mix with existing data.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty dataset directory: {output}")
    with launch_simulation(cfg, {"headless": args.headless, "livestream": 0}):
        import gymnasium as gym
        import torch
        from isaaclab.utils.io import dump_yaml
        from peginsert.tasks.pegact.demonstrations import DemonstrationWriter
        from peginsert.tasks.pegact.expert import ExpertCfg, PegInsertExpert

        env = gym.make("PegInset", cfg=cfg).unwrapped
        writer = None
        interrupted = True
        started = time.monotonic()
        steps = 0
        try:
            with torch.inference_mode():
                observation, _ = env.reset(seed=args.seed)
                expert = PegInsertExpert(env, ExpertCfg(insertion_step=args.insertion_step))
                try:
                    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
                except (OSError, subprocess.CalledProcessError):
                    revision = "unknown"
                metadata = {
                    "task": "PegInset", "physics": "newton_mjwarp", "renderer": "none" if args.headless else "newton_warp",
                    "visualizer": "none" if args.headless else "newton_gl", "kit": False,
                    "reset_strategy": args.reset_strategy, "seed": args.seed, "git_revision": revision,
                    "num_envs": args.num_envs, "state_table_size": args.state_table_size,
                    "step_dt": env.step_dt, "physics_dt": cfg.sim.dt, "decimation": cfg.decimation,
                    "policy_observation_shape": list(observation["policy"].shape[1:]),
                    "history_length": cfg.observations.policy.history_length,
                    "history_order": cfg.observations.policy.history_order,
                    "observation_terms": env.observation_manager.active_terms["policy"],
                    "expert": asdict(expert.cfg), **expert.metadata(),
                }
                writer = DemonstrationWriter(output, env.num_envs, metadata,
                                             keep_failures=args.keep_failures, validation_fraction=args.validation_fraction)
                dump_yaml(str(output / "env.yaml"), cfg)
                print(f"[expert] dataset={output.resolve()} | Newton/MJWarp | Kit=False", flush=True)
                while writer.completed < args.episodes and env.sim.is_running():
                    if args.max_steps is not None and steps >= args.max_steps:
                        break
                    # Clone before step: env returns/reset buffers may alias the old ones.
                    before = observation["policy"].clone()
                    action, diagnostics = expert.act()
                    diagnostics["expert_failure"] = expert.failure_reason.clone()
                    observation, reward, terminated, truncated, _ = env.step(action)
                    manager = env.termination_manager
                    success = manager.get_term("success").clone()
                    failure = torch.zeros_like(success)
                    for name in manager.active_terms:
                        if name != "success" and not manager.get_term_cfg(name).time_out:
                            failure |= manager.get_term(name)
                    diagnostics["failure"] = failure
                    records = writer.append(before, action, reward, terminated, truncated, success,
                                            diagnostics, limit=args.episodes)
                    ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                    expert.reset(ids)
                    steps += 1
                    for record in records:
                        print(f"[expert] episode={record['episode_id']} success={record['success']} "
                              f"length={record['length']} return={record['return']:.3f} "
                              f"saved={writer.saved} success_rate={writer.successes / writer.completed:.1%}", flush=True)
                    if steps % 25 == 0:
                        print(f"[expert] step={steps} stages={torch.bincount(expert.stage, minlength=7).tolist()}", flush=True)
                interrupted = writer.completed < args.episodes
        except KeyboardInterrupt:
            print("[expert] Interrupted; preserving complete episodes only.", flush=True)
        finally:
            if writer is not None:
                summary = writer.close(interrupted=interrupted)
                print(json.dumps({**summary, "vector_steps": steps, "elapsed_s": round(time.monotonic() - started, 2)}, indent=2), flush=True)
            env.close()
        if writer is not None and writer.successes == 0:
            print("[expert] No successful demonstrations collected; this run is not a usable success-only BC dataset.", flush=True)


if __name__ == "__main__":
    main()
