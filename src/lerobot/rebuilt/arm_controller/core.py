"""ArmController module — unified interface for robot arm control.

Provides ArmState (joint + EE pose) and ArmAction (joint / cartesian / hybrid control).
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any


class ControlMode(enum.Enum):
    """Control mode for sending actions to the arm."""

    JOINT = "joint"          # Direct joint position control
    CARTESIAN = "cartesian"   # End-effector pose control (requires IK)
    HYBRID = "hybrid"         # Both joint and EE pose provided


@dataclass
class ArmState:
    """Robot arm state — contains both joint-level and Cartesian information."""

    joint_positions: dict[str, float]    # e.g. {"shoulder_pan": 12.5, "shoulder_lift": -20.0, ...}
    joint_velocities: dict[str, float] | None = None
    ee_pose: dict[str, float] | None = None
        # e.g. {"x": 0.4, "y": 0.1, "z": 0.3,
        #        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
        # (quaternion [q_w, q_x, q_y, q_z] for rotation)
    ee_velocity: dict[str, float] | None = None
    timestamp: float | None = None

    def to_robot_observation(self) -> dict[str, Any]:
        """Convert to legacy RobotObservation dict format ({motor}.pos)."""
        obs: dict[str, Any] = {}
        for name, val in self.joint_positions.items():
            obs[f"{name}.pos"] = val
        if self.ee_pose is not None:
            for k, v in self.ee_pose.items():
                obs[f"ee.{k}"] = v
        return obs


@dataclass
class ArmAction:
    """Robot arm action — supports joint, cartesian, and hybrid control modes."""

    joint_positions: dict[str, float] | None = None   # Joint-space control
    ee_pose: dict[str, float] | None = None            # Cartesian-space control
        # e.g. {"x": ..., "y": ..., "z": ...,
        #        "qw": ..., "qx": ..., "qy": ..., "qz": ...}
        # (quaternion [q_w, q_x, q_y, q_z] for rotation)
    control_mode: ControlMode = ControlMode.JOINT

    def to_robot_action(self) -> dict[str, Any]:
        """Convert to legacy RobotAction dict format ({motor}.pos)."""
        action: dict[str, Any] = {}
        if self.joint_positions is not None:
            for name, val in self.joint_positions.items():
                action[f"{name}.pos"] = val
        if self.ee_pose is not None:
            for k, v in self.ee_pose.items():
                action[f"ee.{k}"] = v
        return action

    @classmethod
    def from_robot_action(cls, action: dict[str, Any]) -> ArmAction:
        """Create ArmAction from legacy RobotAction dict."""
        joint_positions: dict[str, float] = {}
        ee_pose: dict[str, float] = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                joint_positions[key.removesuffix(".pos")] = float(val)
            elif key.startswith("ee."):
                ee_pose[key[3:]] = float(val)

        control_mode = ControlMode.JOINT
        if ee_pose and not joint_positions:
            control_mode = ControlMode.CARTESIAN
        elif ee_pose and joint_positions:
            control_mode = ControlMode.HYBRID

        return cls(
            joint_positions=joint_positions if joint_positions else None,
            ee_pose=ee_pose if ee_pose else None,
            control_mode=control_mode,
        )


class ArmController(abc.ABC):
    """Unified abstract interface for robot arm control.

    Subclasses must implement all abstract methods. The interface decouples
    the arm control from cameras and data collection, providing a clean
    boundary for the refactored architecture.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this arm controller type."""
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the arm is currently connected."""
        ...

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """Whether the arm is currently calibrated."""
        ...

    @property
    @abc.abstractmethod
    def motor_names(self) -> list[str]:
        """Ordered list of motor/joint names."""
        ...

    @property
    @abc.abstractmethod
    def action_features(self) -> dict[str, type]:
        """Feature dict for actions (legacy compatibility)."""
        ...

    @property
    @abc.abstractmethod
    def observation_features(self) -> dict[str, type | tuple]:
        """Feature dict for observations (legacy compatibility, excluding cameras)."""
        ...

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the arm hardware."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the arm hardware."""
        ...

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Run calibration procedure."""
        ...

    @abc.abstractmethod
    def configure(self) -> None:
        """Apply motor configuration (PID, torque limits, etc.)."""
        ...

    @abc.abstractmethod
    def get_state(self) -> ArmState:
        """Read current state (joint positions + EE pose via FK)."""
        ...

    @abc.abstractmethod
    def send_action(self, action: ArmAction) -> ArmAction:
        """Send action command, return the actually executed action.

        If control_mode is CARTESIAN, IK is performed internally.
        If control_mode is JOINT, action is sent directly.
        If control_mode is HYBRID, joint_positions are used but ee_pose is
        also available for logging.
        """
        ...

    @abc.abstractmethod
    def forward_kinematics(self, joint_positions: dict[str, float]) -> dict[str, float]:
        """Forward kinematics: joint positions → EE pose dict."""
        ...

    @abc.abstractmethod
    def inverse_kinematics(
        self, ee_pose: dict[str, float], current_joints: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Inverse kinematics: EE pose → joint positions."""
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
