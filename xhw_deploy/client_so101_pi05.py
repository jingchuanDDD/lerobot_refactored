#!/usr/bin/env python3
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
SO101 Real Robot Client for OpenPi Inference Server.

Connects to the WebSocket server started by scripts/run_server_so101.sh
and runs the SO101 robot in a closed-loop control loop.

Usage:
    python examples/client_so101.py \
        --server-host localhost \
        --server-port 8885 \
        --robot-port /dev/ttyACM1 \
        --camera-external 0 \
        --camera-wrist 2 \
        --task "Put the lemon into the fruit basket."
"""

import argparse
import logging
import signal
import time

import numpy as np
import cv2

# openpi_client is installed as part of the openpi package
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# LeRobot SO101 hardware driver
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_NAME = "gripper"

def _parse_image_for_network(img: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> bytes:
    """
    Format, resize (with pad), and compress the image for network transmission.
    Fully aligns with openpi training transforms: 224x224, resize_with_pad, bilinear.
    """
    # 1. 确保图像格式安全 (保留你原有的防错逻辑)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    target_w, target_h = target_size
    cur_h, cur_w = img.shape[:2]
    
    # 2. 等比缩放计算 (严格对齐 openpi_client.image_tools.resize_with_pad 逻辑)
    # 计算缩放比例: ratio = max(cur_width / width, cur_height / height)
    ratio = max(cur_w / target_w, cur_h / target_h)
    new_w = max(1, int(cur_w / ratio))
    new_h = max(1, int(cur_h / ratio))
    
    # 使用 Bilinear 双线性插值进行缩放
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. 创建 224x224 纯黑背景 (Padding)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中贴图的 offset
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    # 将缩放后的图像 Paste 到黑底图上
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img
    
    # 4. JPEG 压缩 (质量 85，极大降低 WebSocket 传输体积防阻塞)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, encoded_image = cv2.imencode('.jpg', canvas, encode_param)
    
    return encoded_image.tobytes()

def _parse_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 HWC."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


class SO101Controller:
    """Closed-loop controller for SO101 using the openpi WebSocket server."""

    def __init__(
        self,
        server_host: str,
        server_port: int,
        robot_port: str,
        camera_external: int,
        camera_wrist: int,
        task: str,
        control_fps: float = 10.0,
        camera_width: int = 640,
        camera_height: int = 360,
        camera_fps: int = 30,
        max_relative_target: float | None = None,
        chunk_execute_steps: int = 10,
    ):
        self.task = task
        self.control_fps = control_fps
        self.chunk_execute_steps = chunk_execute_steps
        self._running = False

        # openpi WebSocket client — handles connection + metadata handshake:client_server_less_info 224*224,
        logger.info(f"Connecting to openpi server at ws://{server_host}:{server_port} ...")
        self.policy = WebsocketClientPolicy(host=server_host, port=server_port)
        logger.info(f"Server metadata: {self.policy.get_server_metadata()}")

        # SO101 robot
        robot_config = SOFollowerRobotConfig(
            id="pzj_follower_arm",
            port=robot_port,
            use_degrees=True,
            max_relative_target=max_relative_target,
            cameras={
                "external": OpenCVCameraConfig(
                    index_or_path=camera_external, fps=camera_fps, width=640, height=480
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=camera_wrist, fps=camera_fps, width=320, height=240
                ),
            },
        )
        self.robot = SOFollower(robot_config)

    def connect(self):
        logger.info("Connecting to SO101 robot...")
        self.robot.connect()
        logger.info("Robot connected.")

    def disconnect(self):
        logger.info("Disconnecting...")
        try:
            self.robot.disconnect()
        except Exception:
            pass
        logger.info("Done.")

    def _read_obs(self) -> dict:
        """Read robot state + images and format for SO101Inputs.

        SO101Inputs expects:
            observation/image       : HWC uint8  (external camera)
            observation/wrist_image : HWC uint8  (wrist camera)
            observation/state       : (6,) float32  joint positions in degrees
            prompt                  : str
        """
        raw = self.robot.get_observation()

        joint_pos = np.array(
            [raw[f"{n}.pos"] for n in JOINT_NAMES] + [raw[f"{GRIPPER_NAME}.pos"]],
            dtype=np.float32,
        )  # shape (6,)

        return {
            "observation/image": _parse_image_for_network(raw["external"]),
            "observation/wrist_image": _parse_image_for_network(raw["wrist"]),
            "observation/state": joint_pos,
            "prompt": self.task,
        }

    def _execute_chunk(self, result: dict):
        """Execute the action chunk returned by the server.

        SO101Outputs returns {"actions": (T, 6)} in degrees.
        """
        actions = np.asarray(result["actions"])  # (T, 6)
        steps = min(self.chunk_execute_steps, len(actions))
        dt = 1.0 / self.control_fps

        for i in range(steps):
            if not self._running:
                break
            t0 = time.perf_counter()

            cmd = {f"{n}.pos": float(actions[i, j]) for j, n in enumerate(JOINT_NAMES)}
            cmd[f"{GRIPPER_NAME}.pos"] = float(actions[i, 5])
            self.robot.send_action(cmd)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def run(self, max_steps: int = 1000):
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_running", False))

        logger.info(
            f"Starting control loop | task='{self.task}' | "
            f"fps={self.control_fps} | chunk_steps={self.chunk_execute_steps}"
        )
        logger.info("Press Ctrl+C to stop.")

        step = 0
        while self._running and step < max_steps:
            t0 = time.perf_counter()

            obs = self._read_obs()

            t_infer = time.perf_counter()
            result = self.policy.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1000

            self._execute_chunk(result)

            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"Step {step:4d} | infer={infer_ms:.0f}ms | total={total_ms:.0f}ms")
            step += 1

        logger.info(f"Stopped after {step} steps.")


def main():
    p = argparse.ArgumentParser(description="SO101 client for openpi WebSocket server")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8785)
    p.add_argument("--robot-port", default="/dev/ttyACM1")
    p.add_argument("--camera-external", type=int, default=10)
    p.add_argument("--camera-wrist", type=int, default=4)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--task_idx", required=True)
    p.add_argument("--control-fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--chunk-execute-steps", type=int, default=10)
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="Safety limit: max joint change per step (degrees)")

    args = p.parse_args()

    
    task_choices = ["Pick the cube into the plate."]
    task = task_choices[int(args.task_idx)]

    ctrl = SO101Controller(
        server_host=args.server_host,
        server_port=args.server_port,
        robot_port=args.robot_port,
        camera_external=args.camera_external,
        camera_wrist=args.camera_wrist,
        task=task,
        control_fps=args.control_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
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
