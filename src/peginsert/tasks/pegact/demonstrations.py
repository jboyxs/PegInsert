"""Episode-oriented BC writer. No post-reset observation is used as a label."""

import hashlib
import json
from pathlib import Path

import torch


class DemonstrationWriter:
    """Stream complete episodes to disk, keeping only one vector rollout in memory.

    ``policy_observation[t]`` is the raw five-frame actor input *before*
    ``action[t]``. Normalization belongs to the subsequently trained actor.
    All tensors are CPU tensors loadable with ``torch.load(weights_only=True)``.
    """

    def __init__(self, directory, num_envs, metadata, *, keep_failures=False, validation_fraction=0.1):
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must lie in [0, 1)")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Never mix contracts or overwrite an earlier collection.
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Use an empty dataset directory: {self.directory}")
        self.keep_failures = keep_failures
        self.validation_fraction = validation_fraction
        self.pending = [[] for _ in range(num_envs)]
        self.completed = self.saved = self.successes = self.transitions = 0
        self.metadata = {"format_version": 1, "observation_key": "policy", "normalized": False, **metadata}
        self._write_json("metadata.json", self.metadata)

    def _write_json(self, name, value):
        destination = self.directory / name
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)

    def append(self, observation, action, reward, terminated, truncated, success, diagnostics, *, limit=None):
        tensors = {
            "policy_observation": observation, "action": action, "reward": reward,
            "terminated": terminated, "truncated": truncated, "success": success,
            **diagnostics,
        }
        tensors = {name: value.detach().cpu().clone() for name, value in tensors.items()}
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise FloatingPointError("Non-finite transition; refusing to write BC data")
        finished = []
        for env_id, pending in enumerate(self.pending):
            if limit is not None and self.completed >= limit:
                break
            pending.append({name: value[env_id] for name, value in tensors.items()})
            if bool(tensors["terminated"][env_id] | tensors["truncated"][env_id]):
                finished.append(self._finish(env_id))
        return finished

    def _finish(self, env_id):
        pending = self.pending[env_id]
        episode = {name: torch.stack([step[name] for step in pending]) for name in pending[0]}
        # A success and another failure termination can occur simultaneously.
        success = (bool(episode["success"][-1]) and not bool(episode["failure"][-1])
                   and not bool(episode["expert_failure"].any()))
        episode_id = self.completed
        self.completed += 1
        self.successes += int(success)
        digest = hashlib.sha256(f"{self.metadata.get('seed', 0)}:{episode_id}".encode()).digest()
        split = "validation" if int.from_bytes(digest[:8], "big") / 2**64 < self.validation_fraction else "train"
        record = {
            "episode_id": episode_id, "env_id": env_id, "success": success,
            "length": len(pending), "return": float(episode["reward"].sum()), "split": split,
            "terminated": bool(episode["terminated"][-1]), "truncated": bool(episode["truncated"][-1]),
            "expert_failure": int(episode["expert_failure"].max()), "file": None,
        }
        if success or self.keep_failures:
            folder = self.directory / split
            folder.mkdir(exist_ok=True)
            filename = folder / f"episode_{episode_id:06d}.pt"
            temporary = filename.with_suffix(".tmp")
            torch.save({**episode, "episode_id": episode_id, "episode_success": success}, temporary)
            temporary.replace(filename)
            record["file"] = filename.relative_to(self.directory).as_posix()
            self.saved += 1
            self.transitions += len(pending)
        with (self.directory / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        pending.clear()
        return record

    def close(self, *, interrupted=False):
        summary = {
            "completed_episodes": self.completed, "successful_episodes": self.successes,
            "saved_episodes": self.saved, "saved_transitions": self.transitions,
            "success_rate": self.successes / max(self.completed, 1),
            "discarded_partial_episodes": sum(bool(steps) for steps in self.pending),
            "interrupted": interrupted,
        }
        self._write_json("summary.json", summary)
        return summary
