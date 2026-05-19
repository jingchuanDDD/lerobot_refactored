"""CameraArray — multi-camera perception module using lerobot cameras."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs

from .core import Observation, PerceptionModule

logger = logging.getLogger(__name__)


class CameraArray(PerceptionModule):
    """Manages multiple cameras using the existing lerobot camera infrastructure.

    This module is extracted from the Robot class to decouple camera management
    from arm control.
    """

    def __init__(self, camera_configs: dict[str, Any]):
        """
        Args:
            camera_configs: Dict of camera name -> CameraConfig.
                Same format as robot.config.cameras.
        """
        self._camera_configs = camera_configs
        self._cameras = make_cameras_from_configs(camera_configs)

    @property
    def name(self) -> str:
        return "camera_array"

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self._cameras.values())

    @property
    def cameras(self) -> dict:
        """Access underlying camera objects."""
        return self._cameras

    @property
    def observation_features(self) -> dict[str, tuple]:
        """Return image shapes for each camera."""
        return {
            name: (self._camera_configs[name].height, self._camera_configs[name].width, 3)
            for name in self._cameras
        }

    def connect(self) -> None:
        """Connect all cameras."""
        for name, cam in self._cameras.items():
            start = time.perf_counter()
            cam.connect()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.info(f"Camera '{name}' connected in {dt_ms:.0f}ms")

    def disconnect(self) -> None:
        """Disconnect all cameras."""
        for name, cam in self._cameras.items():
            cam.disconnect()
            logger.info(f"Camera '{name}' disconnected")

    def get_observation(self) -> Observation:
        """Read latest frame from all cameras."""
        images = {}
        timestamps = {}
        for name, cam in self._cameras.items():
            start = time.perf_counter()
            images[name] = cam.read_latest()
            timestamps[name] = time.time()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"Camera '{name}' read: {dt_ms:.1f}ms")

        return Observation(images=images, timestamps=timestamps)
