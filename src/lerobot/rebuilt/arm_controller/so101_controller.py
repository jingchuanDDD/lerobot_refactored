"""SO101 ArmController implementation — wraps SO101Follower (Feetech motors)."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from .core import ArmAction, ArmController, ArmState, ControlMode

logger = logging.getLogger(__name__)

# Default SO101 motor configuration
SO101_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

SO101_MOTOR_NAMES = list(SO101_MOTORS.keys())


class SO101ArmController(ArmController):
    """SO101 arm controller using Feetech motors.

    This is a clean implementation that separates arm control from cameras.
    Cameras are handled by the PerceptionModule instead.
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor] | None = None,
        calibration: dict[str, MotorCalibration] | None = None,
        use_degrees: bool = True,
        max_relative_target: float | None = None,
        disable_torque_on_disconnect: bool = True,
        arm_id: str = "",
        calibration_dir: str | Path | None = None,
    ):
        self._port = port
        self._use_degrees = use_degrees
        self._max_relative_target = max_relative_target
        self._disable_torque_on_disconnect = disable_torque_on_disconnect
        self._id = arm_id

        # Auto-load calibration from standard lerobot path if not provided
        if calibration is None and arm_id:
            calibration = self._load_calibration_from_file(calibration_dir)

        motor_config = motors or SO101_MOTORS.copy()
        if use_degrees:
            for name, m in motor_config.items():
                if name != "gripper":
                    m.norm_mode = MotorNormMode.DEGREES

        self.bus = FeetechMotorsBus(
            port=port,
            motors=motor_config,
            calibration=calibration or {},
        )

        self._kinematics = None  # lazily initialized

    def _load_calibration_from_file(self, calibration_dir: str | Path | None = None) -> dict[str, MotorCalibration | None]:
        """Load calibration from the standard lerobot calibration directory."""
        import os

        if calibration_dir is None:
            calibration_dir = HF_LEROBOT_CALIBRATION / "robots" / "so_follower"
        else:
            calibration_dir = Path(calibration_dir)

        fpath = Path(calibration_dir) / f"{self._id}.json"
        if fpath.is_file():
            try:
                with open(fpath) as f, draccus.config_type("json"):
                    calib = draccus.load(dict[str, MotorCalibration], f)
                logger.info(f"Loaded calibration from {fpath}")
                return calib
            except Exception as e:
                logger.warning(f"Failed to load calibration from {fpath}: {e}")

        logger.warning(f"No calibration file found at {fpath}")
        return {}

    @property
    def name(self) -> str:
        return "so101_arm_controller"

    @property
    def motor_names(self) -> list[str]:
        return list(self.bus.motors.keys())

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("Arm not calibrated, running calibration...")
            self.calibrate()
        self.configure()
        logger.info(f"{self.name} connected on {self._port}")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(self._disable_torque_on_disconnect)
        logger.info(f"{self.name} disconnected")

    def calibrate(self) -> None:
        """Interactive calibration — same procedure as SOFollower."""
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self.name} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.bus.write_calibration(
            {
                motor: MotorCalibration(
                    id=self.bus.motors[motor].id,
                    drive_mode=0,
                    homing_offset=homing_offsets[motor],
                    range_min=range_mins[motor],
                    range_max=range_maxes[motor],
                )
                for motor in self.bus.motors
            }
        )

    def configure(self) -> None:
        """Configure motor parameters (PID, torque limits)."""
        with self.bus.torque_disabled():
            try:
                self.bus.configure_motors()
            except RuntimeError as e:
                logger.warning(f"Motor config failed (non-critical): {e}")
            for motor in self.bus.motors:
                try:
                    self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                    self.bus.write("P_Coefficient", motor, 16)
                    self.bus.write("I_Coefficient", motor, 0)
                    self.bus.write("D_Coefficient", motor, 32)
                    if motor == "gripper":
                        self.bus.write("Max_Torque_Limit", motor, 500)
                        self.bus.write("Protection_Current", motor, 250)
                        self.bus.write("Overload_Torque", motor, 25)
                except RuntimeError as e:
                    logger.warning(f"Config motor '{motor}' failed: {e}")

    @check_if_not_connected
    def get_state(self) -> ArmState:
        """Read current joint positions and compute EE pose via FK."""
        start = time.perf_counter()
        obs = self.bus.sync_read("Present_Position")
        joint_positions = dict(obs)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"Read state: {dt_ms:.1f}ms")

        # Compute EE pose via FK if kinematics available
        ee_pose = None
        if self._kinematics is not None:
            try:
                ee_pose = self.forward_kinematics(joint_positions)
            except Exception as e:
                logger.warning(f"FK computation failed: {e}")

        return ArmState(
            joint_positions=joint_positions,
            ee_pose=ee_pose,
            timestamp=time.time(),
        )

    @check_if_not_connected
    def send_action(self, action: ArmAction) -> ArmAction:
        """Send action to the arm, handling different control modes."""
        if action.control_mode == ControlMode.CARTESIAN and action.ee_pose is not None:
            # IK: convert EE pose to joint positions
            current_state = self.get_state()
            ik_joints = self.inverse_kinematics(
                action.ee_pose, current_joints=current_state.joint_positions
            )
            goal_pos = {k: v for k, v in ik_joints.items() if k in self.bus.motors}
        elif action.control_mode == ControlMode.HYBRID and action.joint_positions is not None:
            # Use provided joint positions, ee_pose is for logging only
            goal_pos = {k: v for k, v in action.joint_positions.items() if k in self.bus.motors}
        elif action.joint_positions is not None:
            # JOINT mode
            goal_pos = {k: v for k, v in action.joint_positions.items() if k in self.bus.motors}
        else:
            raise ValueError("ArmAction must have joint_positions or ee_pose")

        # Safety: clip relative target if configured
        if self._max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            from lerobot.robots.utils import ensure_safe_goal_position

            goal_present_pos = {k: (goal_pos[k], present_pos[k]) for k in goal_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self._max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return ArmAction(
            joint_positions=goal_pos,
            ee_pose=action.ee_pose,
            control_mode=action.control_mode,
        )

    def forward_kinematics(self, joint_positions: dict[str, float]) -> dict[str, float]:
        """Compute EE pose from joint positions using RobotKinematics.

        Returns:
            Dict with position (x,y,z) and quaternion rotation (qw,qx,qy,qz).
        """
        if self._kinematics is None:
            self._init_kinematics()
        if self._kinematics is None:
            raise RuntimeError("Kinematics not available (placo not installed?)")

        # Build joint array in URDF order (SO101_MOTOR_NAMES, includes gripper)
        q = np.array([joint_positions.get(n, 0.0) for n in SO101_MOTOR_NAMES], dtype=float)
        t = self._kinematics.forward_kinematics(q)
        pos = t[:3, 3]
        # Use quaternion for rotation representation
        rot = Rotation.from_matrix(t[:3, :3])
        quat = rot.as_quat()  # [qx, qy, qz, qw]
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "qw": float(quat[3]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
        }

    def inverse_kinematics(
        self,
        ee_pose: dict[str, float],
        current_joints: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute joint positions from EE pose using IK.

        Expects ee_pose with quaternion rotation: x, y, z, qw, qx, qy, qz.
        """
        if self._kinematics is None:
            self._init_kinematics()
        if self._kinematics is None:
            raise RuntimeError("Kinematics not available (placo not installed?)")

        if current_joints is None:
            current_joints = dict(self.bus.sync_read("Present_Position"))

        # Build current joint array in URDF order
        q_curr = np.array(
            [current_joints.get(n, 0.0) for n in SO101_MOTOR_NAMES], dtype=float
        )

        # Build desired transform from quaternion
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_quat(
            np.array([ee_pose["qx"], ee_pose["qy"], ee_pose["qz"], ee_pose["qw"]])
        ).as_matrix()
        t_des[:3, 3] = [ee_pose["x"], ee_pose["y"], ee_pose["z"]]

        q_target = self._kinematics.inverse_kinematics(q_curr, t_des)

        # Return all 6 joints
        result = {}
        for i, name in enumerate(SO101_MOTOR_NAMES):
            result[name] = float(q_target[i])
        return result

    def _init_kinematics(self):
        """Lazily initialize RobotKinematics if placo is available."""
        try:
            from lerobot.model.kinematics import RobotKinematics

            import os
            urdf_path = os.environ.get("SO101_URDF_PATH", "")

            # Auto-discover URDF from project tree (try both full and kinematics-only versions)
            if not urdf_path or not os.path.exists(urdf_path):
                search_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "robot_data", "SO101", "so101_kinematics.urdf"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robot_data", "SO101", "so101_kinematics.urdf"),
                    "robot_data/SO101/so101_kinematics.urdf",
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "robot_data", "SO101", "so101_new_calib.urdf"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robot_data", "SO101", "so101_new_calib.urdf"),
                    "robot_data/SO101/so101_new_calib.urdf",
                ]
                for p in search_paths:
                    abs_p = os.path.abspath(p)
                    if os.path.exists(abs_p):
                        urdf_path = abs_p
                        break

            if urdf_path and os.path.exists(urdf_path):
                logger.info(f"Loading kinematics from URDF: {urdf_path}")
                self._kinematics = RobotKinematics(urdf_path)

                # Log joint names from URDF (for debugging mapping)
                urdf_names = self._kinematics.joint_names
                logger.info(f"URDF joint names (placo order): {urdf_names}")
                match = urdf_names == SO101_MOTOR_NAMES
                logger.info(f"Joint name match with SO101_MOTOR_NAMES: {match}")
            else:
                logger.warning(
                    "URDF not found. FK/IK will not be available. "
                    "Set SO101_URDF_PATH env var or place URDF at robot_data/SO101/"
                )
        except ImportError:
            logger.warning("placo not installed. FK/IK not available.")

    def setup_motors(self) -> None:
        """One-time motor ID setup."""
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
