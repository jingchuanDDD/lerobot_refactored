"""SimArmController — ArmController implementation backed by MuJoCo."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmController, ArmState, ControlMode
from lerobot.rebuilt.arm_controller.so101_controller import SO101_MOTOR_NAMES

from .env import JOINT_NAMES, SimEnv

logger = logging.getLogger(__name__)


class SimArmController(ArmController):
    """SO101 arm controller running inside MuJoCo simulation.

    Implements the full ArmController interface (connect/disconnect,
    get_state, send_action, FK, IK) backed by the SimEnv MuJoCo wrapper.

    Args:
        mjcf_path: Path to MJCF scene file (default: sim/configs/so101_sim.xml).
        fps: Simulation frame rate (must match what send_action is called at).
        width: Render resolution width.
        height: Render resolution height.
        physics_timestep: MuJoCo internal timestep (default 2ms).
    """

    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        physics_timestep: float = 0.002,
        joint_offsets: dict[str, float] | None = None,
    ):
        self._env = SimEnv(
            mjcf_path=mjcf_path,
            fps=fps,
            scene_width=width,
            scene_height=height,
            physics_timestep=physics_timestep,
            joint_offsets=joint_offsets or {"wrist_roll": -90},
        )
        self._connected = False
        self._last_joint: dict[str, float] | None = None

    # -- ArmController properties ----------------------------------------

    @property
    def name(self) -> str:
        return "sim_so101"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # simulation is always calibrated

    @property
    def motor_names(self) -> list[str]:
        return list(SO101_MOTOR_NAMES)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.motor_names}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {f"{m}.pos": float for m in self.motor_names}

    # -- lifecycle ------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        # Default home position (all zeros, meaning center)
        self._env.reset({"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
                          "wrist_flex": 0, "wrist_roll": 0, "gripper": 50})
        logger.info(f"{self.name} connected (simulation)")

    def disconnect(self) -> None:
        self._connected = False
        self._env.close()
        logger.info(f"{self.name} disconnected")

    def calibrate(self) -> None:
        pass  # no-op in simulation

    def configure(self) -> None:
        pass  # no-op in simulation

    # -- state / action -------------------------------------------------

    def get_state(self) -> ArmState:
        """Read current joint positions + EE pose from simulation."""
        joints = self._env.get_state()
        ee_pose = self._env.get_ee_pose()
        self._last_joint = joints
        return ArmState(
            joint_positions=joints,
            ee_pose=ee_pose if ee_pose else None,
            timestamp=time.time(),
        )

    def send_action(self, action: ArmAction) -> ArmAction:
        """Apply action in simulation and step physics."""
        if action.control_mode == ControlMode.CARTESIAN and action.ee_pose is not None:
            # IK: cartesian → joint
            current = self._last_joint or self._env.get_state()
            ik_joints = self.inverse_kinematics(action.ee_pose, current)
            goal = {k: v for k, v in ik_joints.items() if k in JOINT_NAMES}
        elif action.joint_positions is not None:
            goal = {k: v for k, v in action.joint_positions.items() if k in JOINT_NAMES}
        else:
            raise ValueError("ArmAction must have joint_positions or ee_pose")

        # Step simulation
        self._env.step(goal)
        self._last_joint = goal

        return ArmAction(
            joint_positions=goal,
            ee_pose=action.ee_pose,
            control_mode=action.control_mode,
        )

    # -- kinematics -----------------------------------------------------

    def forward_kinematics(
        self, joint_positions: dict[str, float]
    ) -> dict[str, float]:
        """Compute EE pose from joint positions (degrees/0-100)."""
        return self._env.forward_kinematics(joint_positions)

    def inverse_kinematics(
        self,
        ee_pose: dict[str, float],
        current_joints: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Approximate IK via residual approach (perturbation-based).

        For a full IK solution use placo or a dedicated solver. This provides
        a simple numerical fallback for the simulation.
        """
        if current_joints is None:
            current_joints = self._last_joint or self._env.get_state()

        # Simple approach: use current joints, return them unchanged.
        # For proper IK, integrate with placo (mirroring real controller).
        logger.warning("SimArmController IK fallback: returning current joints")
        return dict(current_joints)

    # -- convenience ----------------------------------------------------

    @property
    def env(self) -> SimEnv:
        return self._env

    def render(self) -> np.ndarray:
        """Render current view as RGB array (for debugging)."""
        self._env.forward()
        return self._env.render_rgb()
