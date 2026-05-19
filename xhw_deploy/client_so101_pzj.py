#!/usr/bin/env python3
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
SO101 Real Robot Client for Being-H Inference Server.

Self-contained: only depends on lerobot, zmq, torch, numpy.
No Being-H imports required.

Usage:
    export PYTHONPATH=/path/to/lerobot/src:$PYTHONPATH

    python BeingH/deploy/client_so101.py \
        --server-host localhost \
        --server-port 8885 \
        --robot-port /dev/ttyUSB0 \
        --camera-external 0 \
        --camera-wrist 2 \
        --task "Pick the cube into the plate."
"""

import argparse
import logging
import signal
import sys
import time
from io import BytesIO

import numpy as np
import torch
import zmq
import cv2

# ============================================================
# LeRobot imports (SO101 hardware driver)
# ============================================================
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_NAME = "gripper"


# ============================================================
# Lightweight ZMQ client (replaces BeingHInferenceClient)
# ============================================================
class InferenceClient:
    """Minimal ZMQ client compatible with Being-H inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def _send(self, data: dict) -> dict:
        buf = BytesIO()
        torch.save(data, buf)
        self.socket.send(buf.getvalue())
        resp_buf = BytesIO(self.socket.recv())
        resp = torch.load(resp_buf, weights_only=False)
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._send({"endpoint": "ping"})
            return True
        except Exception:
            return False

    def get_action(self, observations: dict) -> dict:
        state = observations.get("state.joint_position")
        print(f"[CLIENT send] state.joint_position={state}", flush=True)
        return self._send({"endpoint": "get_action", "data": observations})

    def close(self):
        self.socket.close()
        self.ctx.term()


# ============================================================
# SO101 Controller
# ============================================================
class SO101Controller:
    """
    Controls SO101 robot using Being-H VLA model predictions.

    Data flow:
        Robot (degrees) -> Server (degrees) -> action (degrees) -> Robot (degrees)
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        robot_port: str,
        camera_external: int | str,
        camera_wrist: int | str,
        task: str,
        control_fps: float = 30.0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        max_relative_target: float | None = None,
        chunk_execute_steps: int = 16,
    ):
        self.task = task
        self.control_fps = control_fps
        self.chunk_execute_steps = chunk_execute_steps
        self._running = False

        # Inference client
        self.client = InferenceClient(host=server_host, port=server_port)

        # SO101 robot with 2 cameras
        robot_config = SOFollowerRobotConfig(
            id="pzj_follower_arm",
            port=robot_port,
            use_degrees=True,
            max_relative_target=max_relative_target,
            cameras={
                "view1": OpenCVCameraConfig(
                    index_or_path=camera_external, fps=camera_fps, width=640, height=480
                ),
                "view2": OpenCVCameraConfig(
                    index_or_path=camera_wrist, fps=camera_fps, width=320, height=240
                ),
            },
        )
        self.robot = SOFollower(robot_config)

    def connect(self):
        logger.info("Connecting to SO101 robot...")
        self.robot.connect()
        logger.info("Robot connected.")
        logger.info("Pinging Being-H server...")
        if self.client.ping():
            logger.info("Server is alive.")
        else:
            raise ConnectionError("Cannot reach Being-H server.")

    def disconnect(self):
        logger.info("Disconnecting...")
        self.robot.disconnect()
        self.client.close()
        logger.info("Done.")

    def _read_observation(self) -> dict:
        """Read from robot, convert to Being-H format (batched, radians)."""
        raw = self.robot.get_observation()

        joint_deg = np.array(
            [raw[f"{n}.pos"] for n in JOINT_NAMES], dtype=np.float32
        )
        gripper_deg = np.array([raw[f"{GRIPPER_NAME}.pos"]], dtype=np.float32)

        state = np.concatenate([joint_deg, gripper_deg])
        
        front_rgb = cv2.cvtColor(raw["view1"], cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(raw["view2"], cv2.COLOR_BGR2RGB)
        return {
            "video.front_view": np.expand_dims(raw["view1"], axis=0),
            #"video.external": np.expand_dims(raw["view1"], axis=0),
            "video.wrist_view": np.expand_dims(raw["view2"], axis=0),
            "state.joint_position": state.reshape(1, 6),
            "language.instruction": [self.task],
        }

    def _execute_action(self, result: dict):
        """Execute server action (degrees) on robot."""
        # action.joint_position is (1, 16, 6): batch x chunk x (5 joints + gripper), in degrees
        action_deg = np.array(result["action.joint_positions"])
        if action_deg.ndim == 3:
            action_deg = action_deg[0]  # (16, 6)

        steps = min(self.chunk_execute_steps, len(action_deg))
        dt = 1.0 / self.control_fps

        for i in range(steps):
            if not self._running:
                break

            t0 = time.perf_counter()

            action = {f"{n}.pos": float(action_deg[i, j]) for j, n in enumerate(JOINT_NAMES)}
            action[f"{GRIPPER_NAME}.pos"] = float(action_deg[i, 5])
            self.robot.send_action(action)

            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def run(self, max_steps: int = 1000):
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: setattr(self, '_running', False))

        logger.info(f"Control loop started (task='{self.task}', fps={self.control_fps}, "
                     f"chunk_steps={self.chunk_execute_steps})")
        logger.info("Press Ctrl+C to stop.\n")

        step = 0
        while self._running and step < max_steps:
            t0 = time.perf_counter()

            obs = self._read_observation()
            # 将 obs save 为 image
            for key, value in obs.items():
                if key.startswith("video."):
                    img_rgb = value[0]
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{key}_step_{step}.png", img_bgr)
                else:
                    print(f"{key}: {value}")

            t_infer = time.perf_counter()
            result = self.client.get_action(obs)
            print(result)
            infer_ms = (time.perf_counter() - t_infer) * 1000

            self._execute_action(result)

            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"Step {step}: infer={infer_ms:.0f}ms total={total_ms:.0f}ms")
            step += 1

        logger.info(f"Stopped after {step} steps.")


def main():
    p = argparse.ArgumentParser(description="SO101 Robot Client for Being-H")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8765)
    p.add_argument("--robot-port", default="/dev/ttyACM1")
    p.add_argument("--camera-external", type=int, default=8, help="External camera index (video.view1)")
    p.add_argument("--camera-wrist", type=int, default=10, help="Wrist camera index (video.view2)")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--task_idx", required=True)
    p.add_argument("--control-fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--chunk-execute-steps", type=int, default=16)
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="Safety: max joint change per step (degrees)")
    args = p.parse_args()

    task_choices = ["Pick the cube into the plate.", "Stack the cubes to make a tower.", "Put the balls into the plate."]
    task = task_choices[int(args.task_idx)]

    ctrl = SO101Controller(
        server_host=args.server_host, server_port=args.server_port,
        robot_port=args.robot_port,
        camera_external=args.camera_external, camera_wrist=args.camera_wrist,
        task=task, control_fps=args.control_fps,
        camera_width=args.camera_width, camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        max_relative_target=args.max_relative_target,
        chunk_execute_steps=args.chunk_execute_steps,
    )

    try:
        ctrl.connect()
        ctrl.run(max_steps=args.max_steps)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    main()
