"""PerceptionModule — unified interface for sensor observation.

Decouples camera/sensor management from the Robot class.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """Sensor observation data — primarily images from cameras."""

    images: dict[str, np.ndarray] = field(default_factory=dict)
        # {"camera_name": HWC ndarray}
    depth_maps: dict[str, np.ndarray] | None = None
    point_clouds: dict[str, np.ndarray] | None = None
    extra_sensors: dict[str, Any] | None = None
    timestamps: dict[str, float] | None = None

    def to_robot_observation(self) -> dict[str, Any]:
        """Convert to legacy RobotObservation dict format."""
        obs: dict[str, Any] = {}
        for name, img in self.images.items():
            obs[name] = img
        if self.depth_maps:
            for name, depth in self.depth_maps.items():
                obs[f"{name}_depth"] = depth
        return obs

    @classmethod
    def from_robot_observation(cls, obs: dict[str, Any], camera_names: list[str] | None = None) -> Observation:
        """Create Observation from legacy RobotObservation dict.

        Args:
            obs: Legacy observation dict (may contain both motor and camera keys)
            camera_names: If provided, only extract these keys as images
        """
        images = {}
        other = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                if camera_names is None or key in camera_names:
                    images[key] = val
                else:
                    other[key] = val
            elif camera_names and key in camera_names:
                # Allow non-ndarray camera data too
                images[key] = val
            else:
                other[key] = val

        extra = other if other else None
        return cls(images=images, extra_sensors=extra)


class PerceptionModule(abc.ABC):
    """Abstract base class for perception modules.

    Manages sensors (primarily cameras) and outputs Observation data.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifier for this perception module."""
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether all sensors are connected."""
        ...

    @property
    @abc.abstractmethod
    def observation_features(self) -> dict[str, tuple]:
        """Feature dict for observations (image shapes)."""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to all sensors."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from all sensors."""
        ...

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        """Read current sensor data."""
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
