import os

from stable_baselines3.common.vec_env import VecVideoRecorder

from async_video_recorder import AsyncVideoRecorder


class AsyncVecVideoRecorder(VecVideoRecorder):
    video_recorder: AsyncVideoRecorder

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = AsyncVideoRecorder(
            env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True
