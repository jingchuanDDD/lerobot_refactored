"""KeyTeleop — keyboard-driven teleop implementing TeleopModule.

Uses terminal raw mode to read WASD/QE/RF/UJ/TG keys.
Works over SSH without X11.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from functools import cached_property
from typing import Any

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState, ControlMode
from lerobot.rebuilt.arm_controller.so101_controller import SO101_MOTOR_NAMES
from lerobot.rebuilt.teleop_module.core import TeleopModule, TeleopOutput

logger = logging.getLogger(__name__)

# Default joint step sizes (degrees / 0-100 for gripper)
DEFAULT_STEPS = {
    "shoulder_pan": 3.0,
    "shoulder_lift": 3.0,
    "elbow_flex": 3.0,
    "wrist_flex": 3.0,
    "wrist_roll": 5.0,
    "gripper": 10.0,
}

# Key → (joint_name, delta)
KEY_MAP: dict[str, tuple[str, float]] = {
    "w": ("shoulder_lift", 1),
    "s": ("shoulder_lift", -1),
    "a": ("shoulder_pan", 1),
    "d": ("shoulder_pan", -1),
    "q": ("elbow_flex", 1),
    "e": ("elbow_flex", -1),
    "r": ("wrist_flex", 1),
    "f": ("wrist_flex", -1),
    "u": ("wrist_roll", 1),
    "j": ("wrist_roll", -1),
    "t": ("gripper", 1),
    "g": ("gripper", -1),
}


class KeyTeleop(TeleopModule):
    """Keyboard-based teleoperation device.

    Usage::

        teleop = KeyTeleop()
        teleop.connect()
        while True:
            out = teleop.get_action()
            send_to_arm(out.action)
        teleop.disconnect()
    """

    def __init__(
        self,
        steps: dict[str, float] | None = None,
        home: dict[str, float] | None = None,
    ):
        self._steps = steps or DEFAULT_STEPS.copy()
        self._home = home or {n: 0.0 for n in SO101_MOTOR_NAMES}
        self._home["gripper"] = 50.0  # mid-gripper
        self._joints: dict[str, float] = dict(self._home)
        self._connected = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "key_teleop"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in SO101_MOTOR_NAMES}

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        self._connected = True
        self._running = True
        self._joints = dict(self._home)
        self._start_key_reader()
        self._print_help()
        logger.info("KeyTeleop connected")

    def disconnect(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._connected = False
        logger.info("KeyTeleop disconnected")

    def get_action(self, current_state: ArmState | None = None) -> TeleopOutput:
        with self._lock:
            joints = dict(self._joints)
        return TeleopOutput(
            action=ArmAction(
                joint_positions=joints,
                control_mode=ControlMode.JOINT,
            ),
            raw_reading={f"{k}.pos": v for k, v in joints.items()},
            is_intervention=True,
        )

    def send_feedback(self, state: ArmState) -> None:
        pass  # keyboard has no force feedback

    # -- internal ---------------------------------------------------------

    def _start_key_reader(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY — KeyTeleop disabled")
            return

        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            logger.warning("Cannot get terminal settings")
            return

        tty.setraw(sys.stdin.fileno())

        def _read():
            try:
                while self._running:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r:
                        continue
                    ch = os.read(sys.stdin.fileno(), 1)
                    if not ch:
                        break
                    key = ch.decode("utf-8", errors="ignore").lower()
                    self._handle_key(key)
            except Exception:
                pass
            finally:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_read, daemon=True)
        self._thread.start()

    def _handle_key(self, key: str):
        if key == " ":
            # Space = stop (set exit_early via external events)
            return
        if key not in KEY_MAP:
            return
        joint, direction = KEY_MAP[key]
        step = self._steps[joint] * direction
        with self._lock:
            self._joints[joint] += step
            # Clamp to reasonable ranges
            if joint == "gripper":
                self._joints[joint] = max(0, min(100, self._joints[joint]))
            else:
                self._joints[joint] = max(-180, min(180, self._joints[joint]))

    def _print_help(self):
        print("=" * 60)
        print("  KeyTeleop Controls")
        print("=" * 60)
        print("  W/S  : shoulder_lift  ±3°")
        print("  A/D  : shoulder_pan   ±3°")
        print("  Q/E  : elbow_flex     ±3°")
        print("  R/F  : wrist_flex     ±3°")
        print("  U/J  : wrist_roll     ±5°")
        print("  T/G  : gripper        ±10")
        print("  SPACE: exit episode")
        print("=" * 60)
        print("Press WASD to move the arm. SPACE to exit.\n")
