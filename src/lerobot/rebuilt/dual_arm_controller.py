"""DualArmController — wraps real + simulation arm controllers for teleop-in-sim.

Makes unified_loop send the same action to both arms simultaneously.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmController, ArmState, ControlMode

logger = logging.getLogger(__name__)


class DualArmController(ArmController):
    """Composite ArmController that mirrors actions to two controllers.

    Args:
        real: Hardware arm controller (e.g. SO101ArmController).
        sim: Simulation arm controller (SimArmController).
        sync_state: Read state from real (True) or sim (False).
    """

    def __init__(self, real: ArmController, sim: ArmController, sync_state: bool = True):
        self._real = real
        self._sim = sim
        self._sync_state = sync_state

    @property
    def name(self) -> str:
        return f"dual({self._real.name}+{self._sim.name})"

    @property
    def is_connected(self) -> bool:
        return self._real.is_connected and self._sim.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._real.is_calibrated and self._sim.is_calibrated

    @property
    def motor_names(self) -> list[str]:
        return self._real.motor_names

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._real.action_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._real.observation_features

    def connect(self, calibrate: bool = True) -> None:
        self._real.connect(calibrate=calibrate)
        self._sim.connect(calibrate=calibrate)
        logger.info(f"DualArmController connected: real={self._real.name} + sim={self._sim.name}")

    def disconnect(self) -> None:
        self._real.disconnect()
        self._sim.disconnect()

    def calibrate(self) -> None:
        self._real.calibrate()
        # sim is auto-calibrated

    def configure(self) -> None:
        self._real.configure()

    def get_state(self) -> ArmState:
        return self._real.get_state() if self._sync_state else self._sim.get_state()

    def send_action(self, action: ArmAction) -> ArmAction:
        """Send same action to both real and sim arms."""
        real_sent = self._real.send_action(action)
        self._sim.send_action(action)
        return real_sent

    def forward_kinematics(self, joint_positions: dict[str, float]) -> dict[str, float]:
        return self._real.forward_kinematics(joint_positions)

    def inverse_kinematics(
        self, ee_pose: dict[str, float], current_joints: dict[str, float] | None = None
    ) -> dict[str, float]:
        return self._real.inverse_kinematics(ee_pose, current_joints)
