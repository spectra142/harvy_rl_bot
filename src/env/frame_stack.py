"""Frame stacking for temporal observations."""

import numpy as np
from collections import deque
from typing import Dict


class FrameStack:
    """Stack last N observations to provide temporal context.

    Maintains a rolling buffer of the last N observations for each
    observation key, concatenating them to give the policy a sense
    of motion and recent history.

    Attributes:
        n_frames: Number of frames to stack.
        frames: Dictionary of deques holding recent observations per key.
    """

    def __init__(self, obs_space, n_frames: int = 4):
        """Initialize frame stack buffers.

        Args:
            obs_space: The observation space (gym.spaces.Dict).
            n_frames: Number of frames to stack.
        """
        self.n_frames = n_frames
        self.frames: Dict[str, deque] = {
            key: deque(maxlen=n_frames) for key in obs_space.spaces.keys()
        }

    def reset(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Initialize frames with the first observation repeated.

        Args:
            obs: The initial observation from env.reset().

        Returns:
            Stacked observation with N copies of the initial frame.
        """
        for key, value in obs.items():
            self.frames[key].clear()
            for _ in range(self.n_frames):
                self.frames[key].append(value)
        return self._get_stacked()

    def step(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add new observation and return stacked frames.

        Args:
            obs: The new observation from env.step().

        Returns:
            Stacked observation with the latest frame appended.
        """
        for key, value in obs.items():
            self.frames[key].append(value)
        return self._get_stacked()

    def _get_stacked(self) -> Dict[str, np.ndarray]:
        """Stack frames, keeping non-voxel modalities at their current shape.

        To avoid reshaping every encoder in the network, we only stack the
        voxel grid temporally (shape ``(n_frames, 11, 11, 7)``). All other
        modalities return the latest frame.
        """
        stacked: Dict[str, np.ndarray] = {}
        for key, frame_deque in self.frames.items():
            frames = list(frame_deque)
            if len(frames) == 0:
                continue
            if key == "voxel_grid":
                stacked[key] = np.stack(frames, axis=0).astype(np.float32)
            else:
                # Non-voxel arrays are not stacked; use the most recent frame.
                stacked[key] = frames[-1].astype(np.float32)
        return stacked
