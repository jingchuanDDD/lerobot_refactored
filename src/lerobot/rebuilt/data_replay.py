"""DataReplay — offline replay of recorded episodes in simulation.

Loads recorded data (parquet) and replays actions through SimArmController
frame by frame. Works without hardware.
"""

from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState, ControlMode
from lerobot.rebuilt.arm_controller.so101_controller import SO101_MOTOR_NAMES

logger = logging.getLogger(__name__)

JOINT_ORDER = list(SO101_MOTOR_NAMES)  # ["shoulder_pan", ..., "gripper"]


@dataclass
class ReplayFrame:
    """A single replayable frame."""
    action: dict[str, float]     # joint_positions in motor units
    state: dict[str, float]      # recorded state (if available)
    timestamp: float | None = None
    index: int = 0


class DataReplay:
    """Loads and replays recorded SO101 arm data through a simulation arm.

    Usage::

        # load
        replay = DataReplay("outputs_recordings_new/test/refactored_record")
        print(f"Found {replay.num_frames} frames in {replay.num_episodes} episodes")

        # connect sim arm
        from lerobot.rebuilt.sim import SimArmController
        arm = SimArmController(); arm.connect()

        # replay episode 0
        replay.replay_episode(arm, episode_idx=0, fps=30)
        arm.disconnect()
    """

    def __init__(self, dataset_path: str | Path):
        self._path = Path(dataset_path)
        self._frames: list[ReplayFrame] = []
        self._episode_ranges: list[tuple[int, int]] = []
        self._load()

    def _load(self):
        """Load data from parquet files."""
        data_dir = self._find_data_dir()
        if not data_dir:
            logger.warning(f"No data found in {self._path}")
            return

        # Read data parquet
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No parquet files found")
            return

        df = pd.read_parquet(parquet_files[0])
        logger.info(f"Loaded {len(df)} frames from {parquet_files[0].name}")

        # Check if this is a refactored dataset (action as list) or legacy
        action_col = self._find_column(df, ["action", "action.state"])
        state_col = self._find_column(df, ["observation.state", "observation.environment_state"])

        ep_col = self._find_column(df, ["episode_index", "episode", "episode_id"])

        if action_col is None:
            # Legacy format: action.xxx.pos columns
            action_motors = {
                c.replace("action.", "").replace(".pos", ""): c
                for c in df.columns if c.startswith("action.") and c.endswith(".pos")
            }
            if not action_motors:
                raise ValueError(f"Cannot find action data in columns: {list(df.columns)}")
        else:
            action_motors = None  # tensor format

        episode_ends = self._find_episode_ends(df, ep_col)
        prev_end = 0
        cur_ep = 0
        for end in episode_ends:
            self._episode_ranges.append((prev_end, end))
            prev_end = end
            cur_ep += 1

        # Parse frames
        for idx in range(len(df)):
            row = df.iloc[idx]

            if action_motors is not None:
                # Legacy format
                action = {m: float(row[c]) for m, c in action_motors.items()}
            else:
                # Tensor format: action is a list/tensor
                act = row[action_col]
                if isinstance(act, np.ndarray):
                    act = act.tolist()
                action = {JOINT_ORDER[i]: float(act[i]) for i in range(min(len(act), len(JOINT_ORDER)))}

            # State (optional)
            state = {}
            if state_col is not None:
                st = row[state_col]
                if isinstance(st, np.ndarray):
                    st = st.tolist()
                if isinstance(st, (list, np.ndarray)):
                    state = {JOINT_ORDER[i]: float(st[i]) for i in range(min(len(st), len(JOINT_ORDER)))}

            self._frames.append(ReplayFrame(
                action=action,
                state=state,
                timestamp=row.get("timestamp", None),
                index=idx,
            ))

        logger.info(f"Parsed {self.num_frames} frames in {self.num_episodes} episodes")

    def _find_data_dir(self) -> Path | None:
        """Find data directory under the dataset path."""
        for candidate in [
            self._path / "data",
            self._path / "data" / "chunk-000",
            self._path,
        ]:
            if candidate.is_dir() and list(candidate.glob("*.parquet")):
                return candidate
        # recursive search
        for f in self._path.rglob("*.parquet"):
            return f.parent
        return None

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _find_episode_ends(df: pd.DataFrame, ep_col: str | None) -> list[int]:
        """Find boundary indices between episodes."""
        if ep_col is not None:
            eps = df[ep_col].values
            ends = []
            for i in range(1, len(eps)):
                if eps[i] != eps[i - 1]:
                    ends.append(i)
            if not ends or ends[-1] != len(eps):
                ends.append(len(eps))
            return ends
        return [len(df)]  # single episode

    # -- properties ----------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_ranges)

    def get_episode_frames(self, episode_idx: int) -> list[ReplayFrame]:
        """Get all frames for a specific episode."""
        if episode_idx >= self.num_episodes:
            raise IndexError(f"Episode {episode_idx} out of range (0-{self.num_episodes - 1})")
        start, end = self._episode_ranges[episode_idx]
        return self._frames[start:end]

    # -- replay --------------------------------------------------------------

    def replay_episode(
        self,
        arm,
        episode_idx: int = 0,
        fps: int = 30,
        display_data: bool = True,
        on_frame: callable | None = None,
    ) -> list[dict[str, float]]:
        """Replay one episode through the given arm controller.

        Args:
            arm: ArmController instance (SimArmController or DualArmController).
            episode_idx: Which episode to replay.
            fps: Replay frame rate.
            display_data: Print state each frame.
            on_frame: Optional callback(state, action, idx) per frame.

        Returns:
            List of recorded joint positions from the simulation.
        """
        frames = self.get_episode_frames(episode_idx)
        if not frames:
            logger.warning(f"Episode {episode_idx} is empty")
            return []

        logger.info(
            "Replaying episode %d/%d: %d frames @ %d fps",
            episode_idx, self.num_episodes - 1, len(frames), fps,
        )

        frame_dt = 1.0 / fps
        trajectory: list[dict[str, float]] = []

        for i, f in enumerate(frames):
            loop_start = time.perf_counter()

            # Send action to arm
            action = ArmAction(
                joint_positions=f.action,
                control_mode=ControlMode.JOINT,
            )
            arm.send_action(action)

            # Read back state
            state = arm.get_state()
            trajectory.append(dict(state.joint_positions))

            if display_data and (i % 10 == 0 or i == len(frames) - 1):
                jp = state.joint_positions
                info = " ".join(f"{k.split('_')[1][:4]}={v:6.1f}" for k, v in jp.items())
                print(f"  [{i:4d}/{len(frames)}] {info}")

            if on_frame:
                on_frame(state, f, i)

            # FPS control
            elapsed = time.perf_counter() - loop_start
            sleep_t = frame_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        logger.info("Episode %d replay complete (%d frames)", episode_idx, len(trajectory))
        return trajectory

    def replay_sequence(
        self,
        arm,
        frames: Iterable[ReplayFrame] | None = None,
        fps: int = 30,
        display_data: bool = True,
        on_frame: callable | None = None,
    ) -> list[dict[str, float]]:
        """Replay a custom sequence of frames.
        
        Args:
            on_frame: Optional callback(state, frame, idx) called after each step.
        """
        if frames is None:
            frames = self._frames
        frames = list(frames)
        if not frames:
            return []

        trajectory = []
        frame_dt = 1.0 / fps

        for i, f in enumerate(frames):
            loop_start = time.perf_counter()
            arm.send_action(ArmAction(
                joint_positions=f.action,
                control_mode=ControlMode.JOINT,
            ))
            s = arm.get_state()
            trajectory.append(dict(s.joint_positions))

            if on_frame:
                on_frame(s, f, i)
            if display_data and i % 10 == 0:
                print(f"  [{i:4d}/{len(frames)}]")

            elapsed = time.perf_counter() - loop_start
            if frame_dt > elapsed:
                time.sleep(frame_dt - elapsed)

        return trajectory
