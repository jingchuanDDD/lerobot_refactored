"""Test data replay in simulation — load dataset, replay through SimArmController.

Usage:
  python tests/test_data_replay.py                                    # replay ep0
  python tests/test_data_replay.py --episode 1                        # replay ep1
  python tests/test_data_replay.py --fps 10 --display                # 10fps + show state
  python tests/test_data_replay.py --all                              # replay all episodes
  python tests/test_data_replay.py --dataset-path /path/to/dataset    # custom dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Data Replay in Simulation")
    parser.add_argument("--dataset-path", type=str,
                        default="outputs_recordings_new/test/refactored_record",
                        help="Path to dataset directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to replay")
    parser.add_argument("--all", action="store_true",
                        help="Replay all episodes back-to-back")
    parser.add_argument("--fps", type=int, default=30,
                        help="Replay frame rate")
    parser.add_argument("--display", action="store_true",
                        help="Print detailed state each frame")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to replay (for quick test)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show MuJoCo render window")
    args = parser.parse_args()

    from lerobot.rebuilt.data_replay import DataReplay
    from lerobot.rebuilt.sim import SimArmController

    # ---- 1. Load dataset ----
    logger.info(f"Loading dataset: {args.dataset_path}")
    replay = DataReplay(args.dataset_path)
    logger.info(f"  {replay.num_frames} frames, {replay.num_episodes} episodes")

    assert replay.num_frames > 0, "No data loaded!"
    assert replay.num_episodes > 0, "No episodes found!"

    # ---- 2. Quick validation: check first/last frame ----
    frames0 = replay.get_episode_frames(0)
    logger.info(f"  Episode 0: {len(frames0)} frames")
    assert len(frames0) > 0, "Episode 0 is empty"
    first_action = frames0[0].action
    assert len(first_action) >= 5, f"Action has {len(first_action)} joints, expected >=5"
    logger.info(f"  First action joints: {list(first_action.keys())}")
    logger.info("  ✓ Dataset loaded and validated")

    # ---- 3. Connect simulation arm ----
    arm = SimArmController()
    arm.connect()

    # ---- 4. Visualization (optional) ----
    viewer_handle = None
    if args.visualize:
        import mujoco
        from mujoco import viewer as mj_viewer
        sim_env = arm.env
        try:
            viewer_handle = mj_viewer.launch_passive(
                sim_env.model, sim_env.data,
                show_left_ui=False, show_right_ui=True,
            )
            logger.info("MuJoCo viewer opened")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    # ---- 5. Replay ----
    episodes_to_replay = range(replay.num_episodes) if args.all else [args.episode]

    for ep in episodes_to_replay:
        if ep >= replay.num_episodes:
            logger.warning(f"Episode {ep} out of range, skipping")
            continue

        frames = replay.get_episode_frames(ep)
        if args.max_frames:
            frames = frames[:args.max_frames]

        logger.info(f"\nReplaying episode {ep}: {len(frames)} frames @ {args.fps} fps")
        
        # Per-frame viewer sync
        on_frame = (lambda s, f, i: viewer_handle.sync()) if viewer_handle else None
        
        trajectory = replay.replay_sequence(
            arm, frames=frames, fps=args.fps, display_data=args.display,
            on_frame=on_frame,
        )

        # Verify trajectory — the arm should have moved by the end
        if len(trajectory) >= 2:
            start_pos = {k: round(v, 1) for k, v in trajectory[0].items()}
            end_pos = {k: round(v, 1) for k, v in trajectory[-1].items()}
            # Check if action commands changed (dataset may have static frames)
            first_act = frames[0].action
            last_act = frames[-1].action
            action_changed = any(
                abs(last_act.get(k, 0) - first_act.get(k, 0)) > 0.1
                for k in first_act
            )
            logger.info(f"  Start: {start_pos}")
            logger.info(f"  End:   {end_pos}")
            logger.info(f"  Action changed: {action_changed}")
            if action_changed:
                diffs = [abs(trajectory[-1].get(k, 0) - trajectory[0].get(k, 0)) for k in trajectory[0]]
                max_diff = max(diffs) if diffs else 0
                logger.info(f"  Max joint delta: {max_diff:.1f}°")
                assert max_diff > 0.05, f"Arm did not move! max_delta={max_diff:.3f}"
            logger.info("  ✓ Trajectory validated")

    # ---- 6. Cleanup ----
    if viewer_handle:
        try: viewer_handle.close()
        except: pass
    arm.disconnect()
    logger.info("\n✓✓✓ Data replay test PASSED ✓✓✓")


if __name__ == "__main__":
    main()
