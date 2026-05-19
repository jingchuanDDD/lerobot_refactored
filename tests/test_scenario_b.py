"""Scenario B — Real hardware + simulation sync teleop test.

Runs unified_loop with:
  - SO101TeleopLeader (real leader arm)
  - DualArmController wrapping:
      - SO101ArmController (real follower)
      - SimArmController (simulation follower)
  - CameraArray (real cameras) or DummyPerception (no cameras)
  - DataCollector (record to dataset)
  - keyboard_listener (control flow)

Usage:
  python tests/test_scenario_b.py --leader-port /dev/ttyACM0 --follower-port /dev/ttyACM1
  
  python tests/test_scenario_b.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 --visualize --duration 30 --display

  # With cameras:
  python tests/test_scenario_b.py --leader-port /dev/ttyACM0 --follower-port /dev/ttyACM1 \
      --cameras '{"wrist":{"type":"opencv","index_or_path":4,"width":320,"height":240,"fps":30}}'
  
  # With recording:
  python tests/test_scenario_b.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
      --record --repo-id test_recording --duration 30 --single-task "pick cube"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scenario B: Hardware + Simulation teleop-in-sim")
    # ---- ports ----
    parser.add_argument("--leader-port", type=str, default=None,
                        help="Leader arm serial port (default: skip real leader)")
    parser.add_argument("--follower-port", type=str, default=None,
                        help="Follower arm serial port (default: skip real arm)")
    parser.add_argument("--leader-id", type=str, default="leader_001",
                        help="Leader calibration ID")
    parser.add_argument("--arm-id", type=str, default="follower_001",
                        help="Follower calibration ID")
    # ---- camera ----
    parser.add_argument("--cameras", type=str, default=None,
                        help="Camera config in JSON")
    # ---- recording ----
    parser.add_argument("--record", action="store_true",
                        help="Enable data recording")
    parser.add_argument("--repo-id", type=str, default="so101_teleop_sim",
                        help="Dataset repo ID")
    parser.add_argument("--single-task", type=str, default="test_scenario_b",
                        help="Task description")
    # ---- loop control ----
    parser.add_argument("--duration", type=float, default=None,
                        help="Max duration (None = ESC to stop)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Loop frequency")
    parser.add_argument("--display", action="store_true",
                        help="Print state each frame")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="Disable keyboard listener")
    parser.add_argument("--visualize", action="store_true",
                        help="Show MuJoCo render window (requires X11/desktop)")
    args = parser.parse_args()

    # ================================================================
    # 1. Keyboard events
    # ================================================================
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    if not args.no_keyboard:
        try:
            from lerobot.rebuilt.keyboard_listener import init_keyboard_listener
            _, events = init_keyboard_listener()
            logger.info("Keyboard listener active: ESC=stop, →=exit, ←=rerecord")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    # ================================================================
    # 2. Teleop (leader arm)
    # ================================================================
    if args.leader_port:
        from lerobot.rebuilt.teleop_module import SO101TeleopLeader
        teleop = SO101TeleopLeader(port=args.leader_port, leader_id=args.leader_id)
        teleop_fk = None
        logger.info(f"Teleop: SO101TeleopLeader on {args.leader_port}")
    else:
        from lerobot.rebuilt.key_teleop import KeyTeleop
        teleop = KeyTeleop()
        teleop_fk = "keyboard"
        logger.info("Teleop: KeyTeleop (no real leader)")

    # ================================================================
    # 3. Arm (real + simulation)
    # ================================================================
    from lerobot.rebuilt.sim import SimArmController
    from lerobot.rebuilt.dual_arm_controller import DualArmController

    if args.follower_port:
        from lerobot.rebuilt.arm_controller import SO101ArmController
        real_arm = SO101ArmController(port=args.follower_port, arm_id=args.arm_id)
        logger.info(f"Real arm: SO101ArmController on {args.follower_port}")
    else:
        real_arm = None
        logger.info("Real arm: NONE (simulation only)")

    sim_arm = SimArmController()
    logger.info("Sim arm: SimArmController")

    if real_arm:
        arm = DualArmController(real=real_arm, sim=sim_arm)
    else:
        arm = sim_arm  # fall back to simulation-only

    # ================================================================
    # 4. Perception (cameras)
    # ================================================================
    if args.cameras:
        import json
        camera_configs = json.loads(args.cameras)
        from lerobot.rebuilt.perception import CameraArray
        perception = CameraArray(camera_configs)
        logger.info(f"Perception: CameraArray ({list(camera_configs.keys())})")
    else:
        from lerobot.rebuilt.dummy_perception import DummyPerception
        perception = DummyPerception()
        logger.info("Perception: DummyPerception (no cameras)")

    # ================================================================
    # 5. Connect everything
    # ================================================================
    logger.info("Connecting modules ...")
    arm.connect(calibrate=False)
    teleop.connect(calibrate=False)
    perception.connect()
    logger.info("All modules connected ✓")

    # ================================================================
    # 6. Data collector (optional)
    # ================================================================
    collector = None
    if args.record:
        from pathlib import Path
        import shutil
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.rebuilt.data_collector import DataCollector

        action_features = teleop.action_features
        obs_features = dict(arm.observation_features)
        obs_features.update(perception.observation_features)

        dataset_features = {
            **hw_to_dataset_features(action_features, "action"),
            **hw_to_dataset_features(obs_features, "observation"),
        }

        dataset_path = Path("outputs_recordings_new") / args.repo_id
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=dataset_path,
            fps=args.fps,
            features=dataset_features,
            robot_type=arm.name,
            use_videos=True,
            vcodec="h264",
        )
        collector = DataCollector(dataset=dataset, fps=args.fps)
        logger.info(f"Recording enabled → {dataset_path}")

    # ================================================================
    # 7. Visualization (optional: MuJoCo viewer window)
    # ================================================================
    viewer_handle = None
    on_frame_callback = None

    if args.visualize:
        import mujoco
        from mujoco import viewer as mj_viewer

        # Get the sim arm's env
        sim_env = sim_arm.env
        try:
            viewer_handle = mj_viewer.launch_passive(
                sim_env.model, sim_env.data,
                show_left_ui=False, show_right_ui=True,
            )
            # Sync callback: push latest state to viewer each frame
            def _vis_callback(state, obs, action):
                viewer_handle.sync()
            on_frame_callback = _vis_callback
            logger.info("MuJoCo viewer window opened (focus it to see the arm)")
        except Exception as e:
            logger.warning(f"Visualization failed (no X11?): {e}")
            if viewer_handle:
                try: viewer_handle.close()
                except: pass
            viewer_handle = None

    # ================================================================
    # 8. Main loop
    # ================================================================
    from lerobot.rebuilt.unified_loop import unified_loop

    print(f"\n{'='*60}")
    print(f"  Scenario B: teleop-in-sim")
    print(f"  arm={arm.name}, teleop={teleop.name}, perception={perception.name}")
    print(f"  fps={args.fps}, duration={args.duration or 'ESC to stop'}, recording={args.record}")
    print(f"  visualize={'ON' if viewer_handle else 'OFF'}")
    print(f"{'='*60}\n")

    try:
        unified_loop(
            arm=arm,
            teleop=teleop,
            perception=perception,
            collector=collector,
            fps=args.fps,
            duration_s=args.duration,
            single_task=args.single_task,
            display_data=args.display,
            events=events,
            on_frame_callback=on_frame_callback,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # ================================================================
    # 9. Cleanup
    # ================================================================
    if viewer_handle:
        try: viewer_handle.close()
        except: pass
    arm.disconnect()
    teleop.disconnect()
    perception.disconnect()
    if collector:
        collector.finalize()
        logger.info(f"Dataset saved: {dataset_path}")
    logger.info("All modules disconnected ✓")


if __name__ == "__main__":
    main()
