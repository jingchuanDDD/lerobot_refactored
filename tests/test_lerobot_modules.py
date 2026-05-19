#!/usr/bin/env python3
r"""LeRobot模块集成测试脚本 — 测试重构后的4个模块：

  1. ArmController  — 机械臂控制模块
  2. TeleopModule   — 遥操模块
  3. PerceptionModule — 感知模块
  4. DataCollector   — 数采模块

测试内容:
  - 单模块测试: 各模块独立创建、连接、读取、断开
  - 集成测试:   遥操→控制→感知→数采 的完整数据流
  - 录制测试:   完整的录制流程

使用方法:
  # 测试机械臂控制模块
  python tests/test_lerobot_modules.py --test arm --follower-port /dev/ttyACM1

  # 测试遥操模块
  python tests/test_lerobot_modules.py --test teleop --leader-port /dev/ttyACM0

  # 测试感知模块(相机)
  python tests/test_lerobot_modules.py --test perception --cameras '{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}'

  # 测试遥操+控制集成
  python tests/test_lerobot_modules.py --test teleop_and_arm --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1

  # 测试完整录制流程
  python tests/test_lerobot_modules.py --test record --follower-port /dev/ttyACM1 --leader-port /dev/ttyACM0 --cameras '{"wrist":{"type":"opencv","index_or_path":4,"width":320,"height":240,"fps":30},"external":{"type":"opencv","index_or_path":10,"width":640,"height":480,"fps":30}}' --duration 20 --episodes 2 --reset-time 5 --fps 30 --single-task "stack the cubes"

  # 运行所有测试
  python tests/test_lerobot_modules.py --test all --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1
"""

import argparse
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Test 1: ArmController 单模块测试
# ============================================================

def test_arm_controller(port: str, arm_id: str = "", duration_s: float = 5.0, fps: int = 30) -> bool:
    """测试ArmController模块: 创建、连接、读取状态、断开"""
    logger.info("=" * 60)
    logger.info("TEST: ArmController 单模块测试")
    logger.info("=" * 60)

    from lerobot.rebuilt.arm_controller import SO101ArmController, ArmAction, ControlMode

    arm = SO101ArmController(port=port, arm_id=arm_id)

    try:
        # 连接
        logger.info("1. 连接机械臂...")
        arm.connect(calibrate=False)  # 跳过校准以加速测试
        logger.info(f"   ✓ 连接成功, is_connected={arm.is_connected}")

        # 读取状态
        logger.info("2. 读取机械臂状态...")
        for i in range(3):
            state = arm.get_state()
            logger.info(
                f"   帧{i}: "
                + ", ".join(f"{k}={v:.1f}" for k, v in list(state.joint_positions.items())[:3])
                + "..."
            )
            time.sleep(0.1)

        # 发送动作 (保持当前位置)
        logger.info("3. 发送保持位置的动作...")
        current_pos = state.joint_positions.copy()
        action = ArmAction(joint_positions=current_pos, control_mode=ControlMode.JOINT)
        sent_action = arm.send_action(action)
        logger.info(f"   ✓ 动作已发送, 实际执行: {list(sent_action.joint_positions.keys())}")

        # 连续读取
        logger.info(f"4. 连续读取 {duration_s}s @ {fps}Hz...")
        frame_count = 0
        errors = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s:
            loop_start = time.perf_counter()
            try:
                state = arm.get_state()
                frame_count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"   读取出错: {e}")

            dt = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - dt, 0)
            time.sleep(sleep_time)

        elapsed = time.perf_counter() - start_time
        logger.info(f"   结果: {frame_count}帧, {errors}次错误, {frame_count/elapsed:.1f} Hz")

        # 断开
        logger.info("5. 断开连接...")
        arm.disconnect()
        logger.info(f"   ✓ 断开成功, is_connected={arm.is_connected}")

        return errors < frame_count * 0.1

    except Exception as e:
        logger.error(f"ArmController测试失败: {e}")
        try:
            arm.disconnect()
        except Exception:
            pass
        return False


# ============================================================
# Test 2: TeleopModule 单模块测试
# ============================================================

def test_teleop_module(port: str, leader_id: str = "", duration_s: float = 5.0, fps: int = 30) -> bool:
    """测试TeleopModule: 创建、连接、读取动作、断开"""
    logger.info("=" * 60)
    logger.info("TEST: TeleopModule 单模块测试")
    logger.info("=" * 60)

    from lerobot.rebuilt.teleop_module import SO101TeleopLeader

    teleop = SO101TeleopLeader(port=port, leader_id=leader_id)

    try:
        # 连接
        logger.info("1. 连接遥操设备...")
        teleop.connect(calibrate=False)
        logger.info(f"   ✓ 连接成功, is_connected={teleop.is_connected}")

        # 读取动作
        logger.info("2. 读取遥操动作...")
        for i in range(3):
            output = teleop.get_action()
            logger.info(
                f"   帧{i}: "
                + ", ".join(f"{k}={v:.1f}" for k, v in list(output.action.joint_positions.items())[:3])
                + "..."
            )
            time.sleep(0.1)

        # 连续读取
        logger.info(f"3. 连续读取 {duration_s}s @ {fps}Hz...")
        frame_count = 0
        errors = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s:
            loop_start = time.perf_counter()
            try:
                output = teleop.get_action()
                frame_count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"   读取出错: {e}")

            dt = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - dt, 0)
            time.sleep(sleep_time)

        elapsed = time.perf_counter() - start_time
        logger.info(f"   结果: {frame_count}帧, {errors}次错误, {frame_count/elapsed:.1f} Hz")

        # 断开
        teleop.disconnect()
        logger.info("   ✓ 断开成功")

        return errors < frame_count * 0.1

    except Exception as e:
        logger.error(f"TeleopModule测试失败: {e}")
        try:
            teleop.disconnect()
        except Exception:
            pass
        return False


# ============================================================
# Test 3: PerceptionModule 单模块测试
# ============================================================

def test_perception_module(camera_configs_json: str, duration_s: float = 5.0, fps: int = 15) -> bool:
    """测试PerceptionModule (CameraArray): 创建、连接、读取图像、断开"""
    logger.info("=" * 60)
    logger.info("TEST: PerceptionModule 单模块测试")
    logger.info("=" * 60)

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.rebuilt.perception import CameraArray, Observation

    # 解析相机配置
    camera_configs = _parse_camera_configs(camera_configs_json)
    if not camera_configs:
        logger.error("未提供相机配置")
        return False

    perception = CameraArray(camera_configs)

    try:
        # 连接
        logger.info("1. 连接相机...")
        perception.connect()
        logger.info(f"   ✓ 连接成功, 相机: {list(perception.cameras.keys())}")

        # 读取图像
        logger.info("2. 读取图像...")
        for i in range(3):
            obs = perception.get_observation()
            for name, img in obs.images.items():
                logger.info(f"   帧{i} {name}: shape={img.shape}, dtype={img.dtype}")
            time.sleep(0.1)

        # 连续读取
        logger.info(f"3. 连续读取 {duration_s}s @ {fps}Hz...")
        frame_count = 0
        errors = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s:
            loop_start = time.perf_counter()
            try:
                obs = perception.get_observation()
                frame_count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"   读取出错: {e}")

            dt = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - dt, 0)
            time.sleep(sleep_time)

        elapsed = time.perf_counter() - start_time
        logger.info(f"   结果: {frame_count}帧, {errors}次错误, {frame_count/elapsed:.1f} Hz")

        # 断开
        perception.disconnect()
        logger.info("   ✓ 断开成功")

        return errors < frame_count * 0.1

    except Exception as e:
        logger.error(f"PerceptionModule测试失败: {e}")
        try:
            perception.disconnect()
        except Exception:
            pass
        return False


# ============================================================
# Test 4: 遥操+控制集成测试
# ============================================================

def test_teleop_and_arm(
    follower_port: str,
    leader_port: str,
    arm_id: str = "",
    leader_id: str = "",
    duration_s: float = 10.0,
    fps: int = 30,
) -> bool:
    """测试遥操→控制的完整数据流: leader读取 → follower执行"""
    logger.info("=" * 60)
    logger.info("TEST: 遥操+控制集成测试")
    logger.info("=" * 60)

    from lerobot.rebuilt.arm_controller import SO101ArmController, ArmAction, ControlMode
    from lerobot.rebuilt.teleop_module import SO101TeleopLeader

    arm = SO101ArmController(port=follower_port, arm_id=arm_id)
    teleop = SO101TeleopLeader(port=leader_port, leader_id=leader_id)

    try:
        # 连接
        logger.info("1. 连接follower和leader...")
        arm.connect(calibrate=False)
        teleop.connect(calibrate=False)
        logger.info("   ✓ 两者均已连接")

        # 遥操循环
        logger.info(f"2. 遥操循环 {duration_s}s @ {fps}Hz...")
        logger.info("   (移动leader臂来控制follower臂)")

        frame_count = 0
        errors = 0
        latencies = []
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s:
            loop_start = time.perf_counter()

            try:
                # 1. 读取follower状态
                state = arm.get_state()

                # 2. 读取leader动作（leader独立读取自身关节）
                teleop_output = teleop.get_action()
                action = teleop_output.action

                # 3. 发送到follower
                sent_action = arm.send_action(action)

                frame_count += 1
                latencies.append((time.perf_counter() - loop_start) * 1e3)

                # 每100帧打印一次状态
                if frame_count % 100 == 0:
                    avg_lat = sum(latencies[-100:]) / len(latencies[-100:])
                    actual_fps = 1.0 / ((time.perf_counter() - loop_start) / 1e3)
                    logger.info(f"   {frame_count}帧, avg延迟={avg_lat:.1f}ms")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"   循环出错: {e}")

            dt = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - dt, 0)
            time.sleep(sleep_time)

        elapsed = time.perf_counter() - start_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        logger.info(f"   结果: {frame_count}帧, {errors}次错误, {frame_count/elapsed:.1f} Hz, avg延迟={avg_latency:.1f}ms")

        # 断开
        teleop.disconnect()
        arm.disconnect()
        logger.info("3. ✓ 已断开连接")

        return errors < frame_count * 0.1

    except Exception as e:
        logger.error(f"遥操+控制测试失败: {e}")
        try:
            teleop.disconnect()
            arm.disconnect()
        except Exception:
            pass
        return False


# ============================================================
# Test 5: 完整录制流程测试
# ============================================================

def test_record(
    follower_port: str,
    leader_port: str,
    camera_configs_json: str,
    arm_id: str = "",
    leader_id: str = "",
    num_episodes: int = 2,
    episode_time_s: float = 5.0,
    reset_time_s: float = 5.0,
    fps: int = 30,
    repo_id: str = "test/refactored_record",
    single_task: str = "test_refactored",
    enable_keyboard: bool = True,
) -> bool:
    """完整录制流程 — 对标 capture_xhw_morning.py.

    Features:
        - 键盘监听: → 提前结束 | ← 重录 | ESC 停止
        - episode 自动保存 & 重置阶段
        - 多 episode 录制
    """
    logger.info("=" * 60)
    logger.info("TEST: 完整录制流程 (4模块集成)")
    logger.info("=" * 60)

    from lerobot.rebuilt.arm_controller import SO101ArmController
    from lerobot.rebuilt.teleop_module import SO101TeleopLeader
    from lerobot.rebuilt.perception import CameraArray
    from lerobot.rebuilt.data_collector import DataCollector
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.robot_utils import precise_sleep

    # 键盘监听
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    if enable_keyboard:
        try:
            from lerobot.rebuilt.keyboard_listener import init_keyboard_listener
            _, events = init_keyboard_listener()
            logger.info("键盘监听已启动: →=退出episode ←=重录 ESC=停止录制")
        except Exception as e:
            logger.warning(f"键盘监听初始化失败: {e}")

    # 创建各模块
    logger.info("1. 创建各模块...")
    arm = SO101ArmController(port=follower_port, arm_id=arm_id)
    teleop = SO101TeleopLeader(port=leader_port, leader_id=leader_id)

    camera_configs = _parse_camera_configs(camera_configs_json)
    perception = CameraArray(camera_configs) if camera_configs else None

    # 连接
    logger.info("2. 连接各模块...")
    arm.connect(calibrate=False)
    teleop.connect(calibrate=False)
    if perception:
        perception.connect()
    logger.info("   ✓ 所有模块已连接")

    # 创建数据集
    logger.info("3. 创建数据集...")
    from lerobot.datasets.utils import hw_to_dataset_features

    action_features = arm.action_features
    obs_features = dict(arm.observation_features)
    if perception:
        obs_features.update(perception.observation_features)

    dataset_features = {
        **hw_to_dataset_features(action_features, "action"),
        **hw_to_dataset_features(obs_features, "observation"),
    }

    import shutil
    from pathlib import Path
    dataset_path = Path("outputs_recordings_new") / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_path,
        fps=fps,
        features=dataset_features,
        robot_type=arm.name,
        use_videos=True,
        vcodec="h264",
    )

    collector = DataCollector(dataset=dataset, fps=fps)

    # 录制循环
    logger.info(f"4. 录制 {num_episodes} 个episode (录制{episode_time_s}s + 重置{reset_time_s}s) @ {fps}Hz")

    episode_idx = 0
    try:
        while episode_idx < num_episodes and not events["stop_recording"]:
            # ==== 录制阶段 ====
            logger.info(f"\n  === Episode {episode_idx+1}/{num_episodes} ===")
            logger.info(f"  录制 {episode_time_s}s ... (→键提前结束, ←键重录)")

            frame_count = 0
            errors = 0
            start_time = time.perf_counter()

            while time.perf_counter() - start_time < episode_time_s:
                if events["exit_early"]:
                    events["exit_early"] = False
                    logger.info(f"  ⏹ Episode 提前结束 ({frame_count} 帧)")
                    break

                loop_start = time.perf_counter()
                try:
                    state = arm.get_state()
                    if perception:
                        observation = perception.get_observation()
                    else:
                        from lerobot.rebuilt.perception.core import Observation
                        observation = Observation()
                    teleop_output = teleop.get_action()
                    sent_action = arm.send_action(teleop_output.action)
                    collector.collect_from_modules(
                        arm_state=state, observation=observation,
                        action=sent_action, single_task=single_task,
                    )
                    frame_count += 1
                    if frame_count % 50 == 0:
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"    {frame_count}帧, {frame_count/elapsed:.1f} Hz")
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"    循环出错: {e}")

                dt = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / fps - dt, 0.0))

            elapsed = time.perf_counter() - start_time

            # ==== 处理录制结果 ====
            if events["rerecord_episode"]:
                logger.info("  ↺ 重录当前 episode (丢弃缓冲区)")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                collector.clear_episode_buffer()
                continue

            # 保存 episode
            collector.save_episode()
            episode_idx += 1
            logger.info(f"  ✓ Episode 已保存: {frame_count}帧, {errors}次错误, {frame_count/elapsed:.1f} Hz")

            # ==== 重置阶段（最后一个 episode 跳过）====
            if not events["stop_recording"] and episode_idx < num_episodes:
                logger.info(f"  --- 重置阶段 {reset_time_s}s ---")
                reset_start = time.perf_counter()
                reset_frames = 0
                while time.perf_counter() - reset_start < reset_time_s:
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
                    try:
                        state = arm.get_state()
                        teleop_output = teleop.get_action()
                        arm.send_action(teleop_output.action)
                        reset_frames += 1
                    except Exception:
                        pass
                    precise_sleep(1.0 / fps)

        # 最终化
        logger.info("5. 最终化数据集...")
        collector.finalize()
        logger.info(f"   ✓ 数据集: {dataset_path}")

    except KeyboardInterrupt:
        logger.info("手动中断")

    finally:
        try:
            teleop.disconnect()
            arm.disconnect()
            if perception:
                perception.disconnect()
            logger.info("   ✓ 所有模块已断开")
        except Exception:
            pass

    return episode_idx > 0


# ============================================================
# Test 6: 数据结构转换测试 (不需要硬件)
# ============================================================

def test_data_structures() -> bool:
    """测试ArmState, ArmAction, Observation, DataFrame的数据转换"""
    logger.info("=" * 60)
    logger.info("TEST: 数据结构转换测试 (无需硬件)")
    logger.info("=" * 60)

    from lerobot.rebuilt.arm_controller.core import ArmAction, ArmState, ControlMode
    from lerobot.rebuilt.perception.core import Observation
    from lerobot.rebuilt.data_collector.core import DataFrame
    import numpy as np

    try:
        # Test ArmState — all 6 standard lerobot joints + quaternion ee_pose
        state = ArmState(
            joint_positions={
                "shoulder_pan": 12.5,
                "shoulder_lift": -20.0,
                "elbow_flex": -30.0,
                "wrist_flex": 15.0,
                "wrist_roll": 45.0,
                "gripper": 50.0,
            },
            ee_pose={"x": 0.3, "y": 0.1, "z": 0.4, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
            timestamp=time.time(),
        )
        obs_dict = state.to_robot_observation()
        logger.info(f"  ArmState → RobotObservation: {list(obs_dict.keys())}")
        assert "shoulder_pan.pos" in obs_dict
        assert "shoulder_lift.pos" in obs_dict
        assert "elbow_flex.pos" in obs_dict
        assert "wrist_flex.pos" in obs_dict
        assert "wrist_roll.pos" in obs_dict
        assert "gripper.pos" in obs_dict
        assert "ee.x" in obs_dict
        assert "ee.qw" in obs_dict
        logger.info("  ✓ ArmState 转换正确 (6关节 + 四元数)")

        # Test ArmAction
        action_joint = ArmAction(
            joint_positions={
                "shoulder_pan": 15.0,
                "shoulder_lift": -18.0,
                "elbow_flex": -28.0,
                "wrist_flex": 17.0,
                "wrist_roll": 48.0,
                "gripper": 60.0,
            },
            control_mode=ControlMode.JOINT,
        )
        action_dict = action_joint.to_robot_action()
        logger.info(f"  ArmAction (JOINT) → RobotAction: {list(action_dict.keys())}")
        assert "shoulder_pan.pos" in action_dict
        assert "gripper.pos" in action_dict
        logger.info("  ✓ ArmAction (JOINT) 转换正确 (6关节)")

        action_hybrid = ArmAction(
            joint_positions={
                "shoulder_pan": 15.0,
                "shoulder_lift": -18.0,
                "elbow_flex": -28.0,
                "wrist_flex": 17.0,
                "wrist_roll": 48.0,
                "gripper": 60.0,
            },
            ee_pose={"x": 0.3, "y": 0.1, "z": 0.4, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
            control_mode=ControlMode.HYBRID,
        )
        action_dict = action_hybrid.to_robot_action()
        logger.info(f"  ArmAction (HYBRID) → RobotAction: {list(action_dict.keys())}")
        assert "shoulder_pan.pos" in action_dict
        assert "ee.x" in action_dict
        assert "ee.qw" in action_dict
        logger.info("  ✓ ArmAction (HYBRID) 转换正确 (6关节 + 四元数)")

        # Test from_robot_action
        legacy_action = {"shoulder_pan.pos": 15.0, "elbow_flex.pos": -28.0, "ee.x": 0.3, "ee.qw": 1.0, "ee.qx": 0.0}
        parsed = ArmAction.from_robot_action(legacy_action)
        logger.info(f"  RobotAction → ArmAction: mode={parsed.control_mode}, joints={list(parsed.joint_positions.keys())}, ee={list(parsed.ee_pose.keys()) if parsed.ee_pose else 'None'}")
        assert parsed.control_mode == ControlMode.HYBRID
        logger.info("  ✓ from_robot_action 转换正确")

        # Test Observation
        obs = Observation(
            images={"top": np.zeros((480, 640, 3), dtype=np.uint8)},
            timestamps={"top": time.time()},
        )
        obs_dict = obs.to_robot_observation()
        logger.info(f"  Observation → RobotObservation: {list(obs_dict.keys())}")
        assert "top" in obs_dict
        logger.info("  ✓ Observation 转换正确")

        # Test DataFrame
        frame = DataFrame(
            state=state,
            observation=obs,
            action=action_hybrid,
            task="test_task",
            timestamp=time.time(),
        )
        logger.info(f"  DataFrame: state joints={list(frame.state.joint_positions.keys())} ({len(frame.state.joint_positions)} joints), obs images={list(frame.observation.images.keys())}")
        logger.info("  ✓ DataFrame 创建正确 (6关节)")

        logger.info("\n  所有数据结构测试通过!")
        return True

    except Exception as e:
        logger.error(f"数据结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 辅助函数
# ============================================================

def _parse_camera_configs(json_str: str) -> dict:
    """解析JSON格式的相机配置"""
    if not json_str:
        return {}

    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"无法解析相机配置JSON: {json_str}")
        return {}

    configs = {}
    for name, cfg in raw.items():
        cam_type = cfg.get("type", "opencv")
        if cam_type == "opencv":
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            configs[name] = OpenCVCameraConfig(
                index_or_path=cfg.get("index_or_path", cfg.get("index", 0)),
                width=cfg.get("width", 640),
                height=cfg.get("height", 480),
                fps=cfg.get("fps", 30),
            )
        elif cam_type == "realsense":
            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
            configs[name] = RealSenseCameraConfig(
                index_or_path=cfg.get("index_or_path", cfg.get("index", 0)),
                width=cfg.get("width", 640),
                height=cfg.get("height", 480),
                fps=cfg.get("fps", 30),
            )
        else:
            logger.warning(f"不支持相机类型: {cam_type}, 跳过")

    return configs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LeRobot模块集成测试脚本")
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["arm", "teleop", "perception", "teleop_and_arm", "record", "data_structures", "all"],
        help="要运行的测试"
    )
    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM0", help="Follower机械臂串口")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM1", help="Leader遥操臂串口")
    parser.add_argument("--arm-id", type=str, default="pzj_follower_arm", help="Follower机械臂calibration ID")
    parser.add_argument("--leader-id", type=str, default="pzj_leader_arm", help="Leader遥操臂calibration ID")
    parser.add_argument("--cameras", type=str, default="", help="相机配置JSON")
    parser.add_argument("--duration", type=float, default=5.0, help="录制/测试持续时间(秒)")
    parser.add_argument("--fps", type=int, default=30, help="目标频率")
    parser.add_argument("--episodes", type=int, default=2, help="录制测试的episode数")
    parser.add_argument("--reset-time", type=float, default=5.0, help="episode间的重置时间(秒)")
    parser.add_argument("--single-task", type=str, default="test_refactored", help="任务描述")
    parser.add_argument("--no-keyboard", action="store_true", help="禁用键盘监听")

    args = parser.parse_args()
    results = {}

    if args.test in ("data_structures", "all"):
        results["data_structures"] = test_data_structures()

    if args.test in ("arm", "all"):
        if args.test == "all" and not results.get("data_structures"):
            logger.warning("数据结构测试未通过，跳过硬件测试")
            sys.exit(1)
        results["arm"] = test_arm_controller(args.follower_port, arm_id=args.arm_id, duration_s=args.duration, fps=args.fps)

    if args.test in ("teleop", "all"):
        results["teleop"] = test_teleop_module(args.leader_port, leader_id=args.leader_id, duration_s=args.duration, fps=args.fps)

    if args.test in ("perception",) and args.cameras:
        results["perception"] = test_perception_module(args.cameras, duration_s=args.duration, fps=args.fps)

    if args.test in ("teleop_and_arm", "all"):
        results["teleop_and_arm"] = test_teleop_and_arm(
            args.follower_port, args.leader_port,
            arm_id=args.arm_id, leader_id=args.leader_id,
            duration_s=args.duration, fps=args.fps,
        )

    if args.test == "record":
        if not args.cameras:
            logger.warning("录制测试需要相机配置 (--cameras)")
        else:
            results["record"] = test_record(
                args.follower_port, args.leader_port, args.cameras,
                arm_id=args.arm_id, leader_id=args.leader_id,
                num_episodes=args.episodes, episode_time_s=args.duration,
                reset_time_s=args.reset_time, fps=args.fps,
                single_task=args.single_task,
                enable_keyboard=not args.no_keyboard,
            )

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
