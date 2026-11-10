"""Validate the frozen checkpoint and restore the real RSL-RL runner on CPU.

This verifies policy, normalization, optimizer and iteration loading without
starting a simulator, allocating GPU memory, or performing a training update.
"""

import hashlib
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


def main():
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / manifest["latest_checkpoint"]
    entry = next(x for x in manifest["checkpoints"] if x["file"] == path.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["iter"] == manifest["saved_iteration"]
    actor = checkpoint["actor_state_dict"]
    dimension = actor["obs_normalizer._mean"].numel()
    last_layer = max(
        (key for key in actor if key.startswith("mlp.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )

    class CpuEnv(VecEnv):
        num_envs = 1
        num_actions = actor[last_layer].shape[0]
        max_episode_length = 350
        episode_length_buf = torch.zeros(1, dtype=torch.long)
        device = "cpu"
        cfg = {}

        def get_observations(self):
            return TensorDict({"policy": torch.zeros(1, dimension)}, batch_size=[1])

        def step(self, actions):
            raise RuntimeError("Verification must not step or train")

    config = yaml.full_load((folder / "params" / "agent.yaml").read_text())
    config["device"] = "cpu"
    runner = OnPolicyRunner(CpuEnv(), config, log_dir=None, device="cpu")
    runner.load(str(path), map_location="cpu")
    assert runner.current_learning_iteration == checkpoint["iter"]
    for key, tensor in checkpoint["actor_state_dict"].items():
        assert torch.equal(runner.alg.actor.state_dict()[key], tensor), key
    for key, tensor in checkpoint["critic_state_dict"].items():
        assert torch.equal(runner.alg.critic.state_dict()[key], tensor), key
    expected_optimizer = checkpoint["optimizer_state_dict"]
    restored_optimizer = runner.alg.optimizer.state_dict()
    assert restored_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert restored_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter, values in expected_optimizer["state"].items():
        for key, value in values.items():
            restored = restored_optimizer["state"][parameter][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored), (parameter, key)
            else:
                assert value == restored, (parameter, key)
    with torch.inference_mode():
        actions = runner.get_inference_policy(device="cpu")(runner.env.get_observations())
    assert torch.isfinite(actions).all()
    print(json.dumps({
        "checkpoint": path.name,
        "iteration": runner.current_learning_iteration,
        "actor_critic_normalization_restored": True,
        "optimizer_restored": True,
        "optimizer_learning_rate": runner.alg.learning_rate,
        "cpu_inference_finite": True,
        "observation_dimension": dimension,
        "action_dimension": runner.env.num_actions,
    }, indent=2))


if __name__ == "__main__":
    main()
