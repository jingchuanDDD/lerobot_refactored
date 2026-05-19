"""TeleopModule — unified interface for teleoperation devices.

Outputs TeleopOutput containing an ArmAction (with both joint_positions and ee_pose).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState


@dataclass
class TeleopOutput:
    """Output from a teleoperation device."""

    action: ArmAction                # Contains both joint_positions and ee_pose
    raw_reading: dict[str, float]     # Raw sensor values (for debugging)
    is_intervention: bool = True      # Whether this is a human intervention


class TeleopModule(abc.ABC):
    """Abstract base class for teleoperation modules.

    Subclasses must implement connect/disconnect/get_action.
    The get_action method returns a TeleopOutput that includes an ArmAction
    with BOTH joint_positions and ee_pose (via FK).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this teleop module type."""
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the teleop device is connected."""
        ...

    @property
    @abc.abstractmethod
    def action_features(self) -> dict[str, type]:
        """Feature dict for actions this teleop can produce."""
        ...

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the teleop device."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the teleop device."""
        ...

    @abc.abstractmethod
    def get_action(self, current_state: ArmState | None = None) -> TeleopOutput:
        """Get the current teleoperation action.

        Args:
            current_state: Current arm state, used for FK computation
                (leader arm reads joint positions → FK → ee_pose).

        Returns:
            TeleopOutput with action containing both joint_positions and ee_pose.
        """
        ...

    @abc.abstractmethod
    def send_feedback(self, state: ArmState) -> None:
        """Send feedback to the teleop device (e.g. force feedback)."""
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def __del__(self):
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass
