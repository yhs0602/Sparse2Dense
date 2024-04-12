import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from gymnasium import logger
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class AsyncVideoRecorder(VideoRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.close_future = asyncio.Future()  # 작업 완료를 위한 Future 객체

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        frame = self.env.render()
        if isinstance(frame, List):
            self.render_history += frame
            frame = frame[-1]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        else:
            with self.lock:
                self.recorded_frames.append(frame)
                logger.debug(f"Captured frame {len(self.recorded_frames)}")

    # Override
    def close(self):
        # `close_async`를 즉시 반환하고 태스크를 스케줄링합니다.
        asyncio.create_task(self._close_async())

    async def _close_async(self):
        if not self.enabled or self._closed:
            self.close_future.set_result(None)  # 이미 닫혔다면 결과 설정
            return

        with self.lock:
            recorded_frames_copy = list(self.recorded_frames)

        if len(recorded_frames_copy) > 0:
            await asyncio.get_running_loop().run_in_executor(
                self.executor, self._save_video, recorded_frames_copy
            )

        self.write_metadata()
        self._closed = True
        self.close_future.set_result(None)  # 작업 완료를 알림

    def _save_video(self, frames):
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(frames, fps=self.frames_per_sec)
        moviepy_logger = None if self.disable_logger else "bar"
        clip.write_videofile(self.path, logger=moviepy_logger)

    async def wait_until_closed(self):
        """이 메서드를 사용하여 비디오 레코더의 모든 작업이 완료될 때까지 외부에서 기다립니다."""
        await self.close_future
