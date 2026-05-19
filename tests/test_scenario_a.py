"""Scenario A — full simulation teleop test.

Runs unified_loop with:
  - KeyTeleop (keyboard WASD → joint control)
  - SimArmController (MuJoCo simulation)
  - DummyPerception (no cameras)
  - keyboard_listener (→/←/ESC for control flow)

Usage:
  python tests/test_scenario_a.py                  # interactive
  python tests/test_scenario_a.py --duration 30     # auto-stop after 30s
  python tests/test_scenario_a.py --display         # print state each frame
"""

from __future__ import annotations

import argparse
import logging
import sys
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scenario A: Keyboard teleop → Sim Arm")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max duration in seconds (None = run until ESC)")
    parser.add_argument("--display", action="store_true",
                        help="Print joint state to console each frame")
    parser.add_argument("--fps", type=int, default=30,
                        help="Control loop frequency")
    args = parser.parse_args()

    from lerobot.rebuilt.sim import SimArmController
    from lerobot.rebuilt.key_teleop import KeyTeleop
    from lerobot.rebuilt.dummy_perception import DummyPerception
    from lerobot.rebuilt.keyboard_listener import init_keyboard_listener
    from lerobot.rebuilt.unified_loop import unified_loop

    # ---- 1. Keyboard events (→=exit episode, ←=rerecord, ESC=stop) ----
    _, events = init_keyboard_listener()

    # ---- 2. Keyboard teleop (WASD/QE/... → joint commands) ----
    teleop = KeyTeleop()
    teleop.connect()

    # ---- 3. Simulation arm ----
    arm = SimArmController()
    arm.connect()

    # ---- 4. Dummy perception (no cameras) ----
    perception = DummyPerception()
    perception.connect()

    # ---- 5. Run unified loop ----
    print("\nStarting unified_loop ...")
    print(f"  arm={arm.name}, teleop={teleop.name}, perception={perception.name}")
    print(f"  fps={args.fps}, duration={args.duration or 'unlimited'}")
    print(f"  Press ESC to stop, → to exit episode\n")

    try:
        unified_loop(
            arm=arm,
            teleop=teleop,
            perception=perception,
            collector=None,  # no recording in this test
            fps=args.fps,
            duration_s=args.duration,
            single_task="teleop_sim_test",
            display_data=args.display,
            events=events,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        arm.disconnect()
        teleop.disconnect()
        perception.disconnect()
        logger.info("All modules disconnected")


if __name__ == "__main__":
    main()
