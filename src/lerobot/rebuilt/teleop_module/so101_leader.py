"""SO101 TeleopLeader — wraps Feetech bus as a leader arm teleoperation device."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import draccus

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState, ControlMode
from lerobot.rebuilt.arm_controller.so101_controller import SO101_MOTORS, SO101_MOTOR_NAMES
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .core import TeleopModule, TeleopOutput

logger = logging.getLogger(__name__)


class SO101TeleopLeader(TeleopModule):
    """SO101 leader arm teleoperation device.

    Reads joint positions from the leader arm and outputs them as TeleopOutput.
    If an ArmController is provided (for FK), also computes ee_pose.
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor] | None = None,
        calibration: dict[str, MotorCalibration] | None = None,
        use_degrees: bool = True,
        fk_fn=None,  # Optional: callable(joint_positions) -> ee_pose dict
        leader_id: str = "",
    ):
        self._port = port
        self._use_degrees = use_degrees
        self._fk_fn = fk_fn

        # Auto-load calibration from standard lerobot path
        if calibration is None and leader_id:
            calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so_leader" / f"{leader_id}.json"
            if calib_path.is_file():
                try:
                    with open(calib_path) as f, draccus.config_type("json"):
                        calibration = draccus.load(dict[str, MotorCalibration], f)
                    logger.info(f"Loaded leader calibration from {calib_path}")
                except Exception as e:
                    logger.warning(f"Failed to load leader calibration: {e}")

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

    @property
    def name(self) -> str:
        return "so101_teleop_leader"

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if calibrate and not self.bus.is_calibrated:
            logger.info("Leader arm not calibrated, running calibration...")
            self.calibrate()
        self.configure()
        logger.info(f"{self.name} connected on {self._port}")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self.name} disconnected")

    def calibrate(self) -> None:
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
        self.bus.disable_torque()
        try:
            self.bus.configure_motors()
        except RuntimeError as e:
            logger.warning(f"Leader motor config failed (non-critical): {e}")
        for motor in self.bus.motors:
            try:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            except RuntimeError as e:
                logger.warning(f"Leader motor {motor} Operating_Mode failed: {e}")

    @check_if_not_connected
    def get_action(self, current_state: ArmState | None = None) -> TeleopOutput:
        """Read leader arm joint positions and optionally compute ee_pose."""
        start = time.perf_counter()
        raw = self.bus.sync_read("Present_Position")
        joint_positions = dict(raw)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"Leader read: {dt_ms:.1f}ms")

        # Compute EE pose if FK function is available
        ee_pose = None
        if self._fk_fn is not None:
            try:
                ee_pose = self._fk_fn(joint_positions)
            except Exception as e:
                logger.warning(f"FK on leader action failed: {e}")

        action = ArmAction(
            joint_positions=joint_positions,
            ee_pose=ee_pose,
            control_mode=ControlMode.HYBRID if ee_pose else ControlMode.JOINT,
        )

        raw_reading = {f"{m}.pos": v for m, v in raw.items()}

        return TeleopOutput(
            action=action,
            raw_reading=raw_reading,
            is_intervention=True,
        )

    def send_feedback(self, state: ArmState) -> None:
        """TODO: Implement force feedback for SO101 leader."""
        raise NotImplementedError("Force feedback not yet implemented for SO101 leader")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
