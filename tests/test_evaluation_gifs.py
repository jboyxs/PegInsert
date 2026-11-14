"""Ensure GIF episode boundaries use pre-reset images and ignore unsuccessful attempts."""

import unittest

import torch

from peginsert.tasks.pegact.evaluation_gifs import SuccessGifRecorder


class GifBoundaryTests(unittest.TestCase):
    def make_recorder(self):
        recorder = object.__new__(SuccessGifRecorder)
        recorder.target = 3
        recorder.saved = []
        recorder.frames = [["initial_0"], ["initial_1"]]
        recorder.terminal_frames = {0: "terminal_0"}
        recorder.capture = lambda: ["reset_0", "next_1"]
        saved_frames = []

        def save(record, frames):
            saved_frames.append(list(frames))
            recorder.saved.append(record)

        recorder._save = save
        return recorder, saved_frames

    def test_success_ends_before_reset_and_other_env_history_continues(self):
        recorder, saved_frames = self.make_recorder()
        recorder.after_step(torch.tensor([True, False]), [{"env_id": 0, "success": True}])
        self.assertEqual(saved_frames, [["initial_0", "terminal_0"]])
        self.assertEqual(recorder.frames, [["reset_0"], ["initial_1", "next_1"]])
        self.assertEqual(recorder.terminal_frames, {})

    def test_failed_or_unassigned_episodes_are_discarded(self):
        for records in ([], [{"env_id": 0, "success": False}]):
            recorder, saved_frames = self.make_recorder()
            recorder.after_step(torch.tensor([True, False]), records)
            self.assertEqual(saved_frames, [])
            self.assertEqual(recorder.frames[0], ["reset_0"])

    def test_capture_stops_after_requested_number_of_successes(self):
        recorder, saved_frames = self.make_recorder()
        recorder.target = 1
        recorder.capture = lambda: self.fail("Must not render after quota is reached")
        recorder.after_step(torch.tensor([True, False]), [{"env_id": 0, "success": True}])
        self.assertEqual(saved_frames, [["initial_0", "terminal_0"]])
        self.assertEqual(recorder.frames, [[], []])
        recorder.after_step(torch.tensor([False, True]), [])


if __name__ == "__main__":
    unittest.main()
