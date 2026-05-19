"""Unified loop — refactored main control loop combining all four modules.

This replaces the monolithic record_loop / teleop_loop with a clean
modular architecture:
  - ArmController → state
  - PerceptionModule → observation
  - TeleopModule → action
  - DataCollector → dataset
"""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmController, ArmState
from lerobot.rebuilt.data_collector.core import DataCollector, DataFrame
from lerobot.rebuilt.perception.core import Observation, PerceptionModule
from lerobot.rebuilt.teleop_module.core import TeleopModule, TeleopOutput

logger = logging.getLogger(__name__)


def unified_loop(
    arm: ArmController,
    teleop: TeleopModule,
    perception: PerceptionModule,
    collector: DataCollector | None = None,
    fps: int = 30,
    duration_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    events: dict | None = None,
    on_frame_callback: Any | None = None,
):
    """Unified main control loop.

    Data flow per frame:
        1. arm.get_state()         → ArmState (joint_positions + ee_pose)
        2. perception.get_observation() → Observation (images)
        3. teleop.get_action(state)     → TeleopOutput (ArmAction with joint + EE)
        4. arm.send_action(action)      → ArmAction (actually sent)
        5. collector.collect(frame)      → DataFrame → Dataset

    Args:
        arm: ArmController instance.
        teleop: TeleopModule instance.
        perception: PerceptionModule instance.
        collector: Optional DataCollector for recording.
        fps: Target loop frequency.
        duration_s: Maximum duration in seconds. None = infinite.
        single_task: Task description for dataset.
        display_data: Whether to print state to console.
        events: Dict with 'exit_early', 'rerecord_episode', 'stop_recording' keys.
        on_frame_callback: Optional callback(state, obs, action) called each frame.
    """
    from lerobot.utils.robot_utils import precise_sleep

    start_time = time.perf_counter()
    frame_count = 0

    while True:
        loop_start = time.perf_counter()

        # Check events
        if events is not None:
            if events.get("exit_early", False):
                events["exit_early"] = False
                break
            if events.get("stop_recording", False):
                break

        # 1. Read arm state
        state = arm.get_state()

        # 2. Read sensor observations
        observation = perception.get_observation()

        # 3. Get teleop action
        teleop_output = teleop.get_action()
        action = teleop_output.action

        # 4. Send action to arm
        sent_action = arm.send_action(action)

        # 5. Collect data (if recording)
        if collector is not None:
            frame = DataFrame(
                state=state,
                observation=observation,
                action=sent_action,
                task=single_task,
                timestamp=time.time(),
            )
            collector.collect(frame, single_task=single_task)

        # 6. Optional callback (for rerun visualization, etc.)
        if on_frame_callback is not None:
            on_frame_callback(state, observation, sent_action)

        # 7. Display
        if display_data:
            _print_state(state, sent_action)

        frame_count += 1

        # FPS control
        dt = time.perf_counter() - loop_start
        sleep_time = 1.0 / fps - dt
        if sleep_time < 0:
            actual_fps = 1.0 / dt
            if frame_count % 100 == 1:
                logger.warning(
                    f"Loop running at {actual_fps:.1f} Hz (target: {fps} Hz)"
                )
        precise_sleep(max(sleep_time, 0.0))

        # Duration check
        if duration_s is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_s:
                break

    elapsed = time.perf_counter() - start_time
    logger.info(f"Loop finished: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} Hz)")


def _print_state(state: ArmState, action: ArmAction):
    """Print current state and action in a compact format."""
    display_len = max(
        max((len(k) for k in state.joint_positions), default=0),
        max((len(k) for k in action.joint_positions or {}), default=0),
    )
    print(f"\n{'─' * (display_len + 20)}")
    print(f"{'JOINT':<{display_len}} | {'STATE':>7} | {'ACTION':>7}")
    if action.joint_positions:
        for name in state.joint_positions:
            s = state.joint_positions.get(name, 0)
            a = action.joint_positions.get(name, 0) if action.joint_positions else 0
            print(f"{name:<{display_len}} | {s:>7.2f} | {a:>7.2f}")

    if state.ee_pose:
        x = state.ee_pose.get("x", 0)
        y = state.ee_pose.get("y", 0)
        z = state.ee_pose.get("z", 0)
        qw = state.ee_pose.get("qw", 0)
        qx = state.ee_pose.get("qx", 0)
        qy = state.ee_pose.get("qy", 0)
        qz = state.ee_pose.get("qz", 0)
        print(f"\nEE Pose: x={x:.3f} y={y:.3f} z={z:.3f} | quat=(qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f})")
