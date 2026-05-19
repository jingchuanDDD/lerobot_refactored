"""DataCollector — unified data collection module.

Collects state (from ArmController), observation (from PerceptionModule),
and action (from TeleopModule or Policy) into a dataset.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState
from lerobot.processor import RobotAction, RobotObservation
from lerobot.rebuilt.perception.core import Observation

logger = logging.getLogger(__name__)


@dataclass
class DataFrame:
    """Single frame of data — the basic unit of data collection."""

    state: ArmState              # From ArmController
    observation: Observation     # From PerceptionModule
    action: ArmAction           # From TeleopModule or Policy (actually sent)
    task: str | None = None
    timestamp: float | None = None

    def to_legacy_frame(
        self,
        dataset_features: dict | None = None,
        single_task: str | None = None,
    ) -> dict[str, Any]:
        """Convert to legacy dataset frame format.

        Combines state, observation, and action into a flat dict
        compatible with LeRobotDataset.add_frame().
        """
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.utils.constants import ACTION, OBS_STR

        frame: dict[str, Any] = {}

        # State + observation → observation part
        obs_dict = self.state.to_robot_observation()
        obs_dict.update(self.observation.to_robot_observation())

        # Action → action part
        action_dict = self.action.to_robot_action()

        if dataset_features is not None:
            observation_frame = build_dataset_frame(dataset_features, obs_dict, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset_features, action_dict, prefix=ACTION)
            frame = {**observation_frame, **action_frame}
        else:
            for k, v in obs_dict.items():
                frame[f"observation.{k}"] = v
            for k, v in action_dict.items():
                frame[f"action.{k}"] = v

        if single_task or self.task:
            frame["task"] = single_task or self.task

        return frame


class DataCollector:
    """Unified data collection module.

    Collects frames from ArmController, PerceptionModule, and TeleopModule,
    and writes them to a LeRobotDataset.
    """

    def __init__(self, dataset: Any, fps: int = 30):
        """
        Args:
            dataset: A LeRobotDataset instance.
            fps: Target frames per second for the collection loop.
        """
        self.dataset = dataset
        self.fps = fps
        self._frame_count = 0
        self._episode_frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def episode_frame_count(self) -> int:
        return self._episode_frame_count

    def collect(self, frame: DataFrame, single_task: str | None = None) -> None:
        """Collect a single frame and write to the dataset.

        Args:
            frame: DataFrame containing state, observation, and action.
            single_task: Task description for this frame.
        """
        legacy_frame = frame.to_legacy_frame(
            dataset_features=self.dataset.features if hasattr(self.dataset, 'features') else None,
            single_task=single_task,
        )
        self.dataset.add_frame(legacy_frame)
        self._frame_count += 1
        self._episode_frame_count += 1

    def save_episode(self) -> None:
        """Save the current episode to the dataset."""
        self.dataset.save_episode()
        self._episode_frame_count = 0

    def clear_episode_buffer(self) -> None:
        """Clear the current episode buffer (e.g., on rerecord)."""
        self.dataset.clear_episode_buffer()
        self._episode_frame_count = 0

    def finalize(self) -> None:
        """Finalize the dataset (flush buffers, encode videos, etc.)."""
        self.dataset.finalize()

    def collect_from_modules(
        self,
        arm_state: ArmState,
        observation: Observation,
        action: ArmAction,
        task: str | None = None,
        single_task: str | None = None,
    ) -> None:
        """Convenience: create DataFrame and collect it."""
        frame = DataFrame(
            state=arm_state,
            observation=observation,
            action=action,
            task=task,
            timestamp=time.time(),
        )
        self.collect(frame, single_task=single_task)
