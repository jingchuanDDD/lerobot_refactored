"""SimPerception — PerceptionModule backed by MuJoCo rendering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import mujoco
from mujoco import mjtObj

from lerobot.rebuilt.perception.core import Observation, PerceptionModule

from .env import SimEnv

logger = logging.getLogger(__name__)


class SimPerception(PerceptionModule):
    """PerceptionModule that renders MuJoCo cameras.

    Args:
        env: SimEnv instance.
        cameras: Dict mapping camera_name → MJCF camera name.
            e.g. {"observation.images.wrist": "wrist_cam"}.
        width: Render width override (default: env width).
        height: Render height override (default: env height).
    """

    def __init__(
        self,
        env: SimEnv,
        cameras: dict[str, str] | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        self._env = env
        self._connected = False

        # Default: use first available camera as "wrist"
        if cameras is None:
            cams = [
                mujoco.mj_id2name(env.model, mjtObj.mjOBJ_CAMERA, i)
                for i in range(env.model.ncam)
            ]
            cameras = {"observation.images.wrist": cams[0]} if cams else {}

        self._camera_map = cameras
        self._width = width
        self._height = height

    @property
    def name(self) -> str:
        return "sim_perception"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def observation_features(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for obs_name in self._camera_map:
            features[obs_name] = (self._height or 480, self._width or 640, 3)
        return features

    def connect(self) -> None:
        self._connected = True
        logger.info(f"{self.name}: {len(self._camera_map)} cameras")

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> Observation:
        """Render all configured cameras and return an Observation."""
        images: dict[str, np.ndarray] = {}
        for obs_name, mj_cam in self._camera_map.items():
            images[obs_name] = self._env.render_rgb(cam_name=mj_cam)
        return Observation(images=images, timestamps={})
