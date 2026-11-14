"""Capture real Newton frames, including terminal poses before automatic reset."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


class SuccessGifRecorder:
    """Keep complete per-env frame buffers and export the first N counted successes."""

    def __init__(self, env, output, count):
        self.env = env
        self.output = Path(output) / "gifs"
        self.output.mkdir(parents=True, exist_ok=True)
        self.target = count
        self.saved = []
        self.terminal_frames = {}
        self.camera = env.scene["evaluation_camera"]
        self.frames = [[frame] for frame in self.capture()]
        self.frames[0][0].save(self.output / "camera_preview.png")
        if np.asarray(self.frames[0][0]).std() < 2:
            raise RuntimeError("Camera produced a blank image; inspect gifs/camera_preview.png")
        self.original_pre_reset = env.recorder_manager.record_pre_reset
        # This lifecycle hook is called after termination checks and before scene reset.
        env.recorder_manager.record_pre_reset = self._pre_reset

    def capture(self):
        from isaaclab.envs.utils.camera_view import camera_rgb_batch

        # Sync transforms without advancing simulation or observation history.
        self.env.sim.forward()
        origins = self.env.scene.env_origins
        wide_target = origins + torch.tensor([0.4, 0.0, 0.18], device=self.env.device)
        wide_eye = origins + torch.tensor([-0.50, -0.70, 0.75], device=self.env.device)
        self.camera.set_world_poses_from_view(wide_eye, wide_target)
        self.camera.update(0.0, force_recompute=True)
        wide = camera_rgb_batch(self.camera, []).cpu().numpy()[..., :3].copy()
        # Fixed hole-relative close-up; its pose is constant within each episode.
        target = self.env.scene["fixed_asset"].data.root_pos_w.torch.clone()
        target[:, 2] += 0.045
        eye = target + torch.tensor([0.18, 0.20, 0.16], device=self.env.device)
        self.camera.set_world_poses_from_view(eye, target)
        self.camera.update(0.0, force_recompute=True)
        close = camera_rgb_batch(self.camera, []).cpu().numpy()[..., :3].copy()
        return [Image.fromarray(np.concatenate((a, b), axis=1).astype(np.uint8)) for a, b in zip(wide, close, strict=True)]

    def _pre_reset(self, env_ids, *args, **kwargs):
        if len(self.saved) < self.target:
            frames = self.capture()
            if env_ids is None:
                ids = range(self.env.num_envs)
            elif isinstance(env_ids, slice):
                ids = range(self.env.num_envs)[env_ids]
            else:
                ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
            for env_id in ids:
                self.terminal_frames[env_id] = frames[env_id]
        return self.original_pre_reset(env_ids, *args, **kwargs)

    def after_step(self, dones, records):
        if len(self.saved) >= self.target:
            return
        by_env = {record["env_id"]: record for record in records}
        done_ids = set(dones.nonzero(as_tuple=False).flatten().tolist())
        for env_id in done_ids:
            if env_id not in self.terminal_frames:
                raise RuntimeError("Missing pre-reset terminal image")
            self.frames[env_id].append(self.terminal_frames.pop(env_id))
            record = by_env.get(env_id)
            if record and record["success"] and len(self.saved) < self.target:
                self._save(record, self.frames[env_id])
            self.frames[env_id] = []
        if len(self.saved) >= self.target:
            self.frames = [[] for _ in self.frames]
            return
        frames = self.capture()
        for env_id, frame in enumerate(frames):
            self.frames[env_id].append(frame)

    def _save(self, record, frames):
        name = f"success_{len(self.saved) + 1:02d}_episode_{record['episode_id']:04d}.gif"
        path = self.output / name
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default(size=20)
        output_frames = []
        for step, frame in enumerate(frames):
            canvas = Image.new("RGB", (1280, 524), (20, 25, 33))
            canvas.paste(frame, (0, 44))
            draw = ImageDraw.Draw(canvas)
            caption = f"Episode {record['episode_id']} | Env {record['env_id']} | t={step * self.env.step_dt:.2f}s"
            if step == len(frames) - 1:
                caption += " | SUCCESS"
            draw.text((14, 10), caption, fill=(239, 245, 250), font=font)
            draw.text((650, 10), "Hole close-up | gravity -9.81 m/s^2", fill=(169, 219, 196), font=font)
            output_frames.append(canvas)
        durations = [round(self.env.step_dt * 1000)] * len(output_frames)
        durations[-1] += 800
        output_frames[0].save(path, save_all=True, append_images=output_frames[1:], duration=durations,
                              loop=0, optimize=False, disposal=2)
        frames[-1].save(path.with_suffix(".png"))
        result = {**record, "file": str(path.resolve()), "frames": len(frames),
                  "duration_s": sum(durations) / 1000, "terminal_captured_before_reset": True}
        self.saved.append(result)
        record["gif"] = str(path.relative_to(self.output.parent))
        print(f"[eval] saved success GIF: {path}", flush=True)

    def close(self):
        self.env.recorder_manager.record_pre_reset = self.original_pre_reset
