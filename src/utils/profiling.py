import time
from collections import deque


class FPSCounter:
    """
    Sliding-window FPS counter.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = deque(maxlen=window)

    def update(self) -> float:
        """
        Call this once per frame; returns current FPS estimate.
        """
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) < 2:
            return 0.0

        total_time = self.timestamps[-1] - self.timestamps[0]
        if total_time <= 0:
            return 0.0

        fps = (len(self.timestamps) - 1) / total_time
        return fps
