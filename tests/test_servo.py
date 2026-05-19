#!/usr/bin/env python3
"""舵机/电机测试脚本 — 独立于lerobot框架，直接测试Feetech电机总线。

测试内容:
  1. 连接测试 — 验证串口通信
  2. Ping扫描 — 探测总线上所有电机
  3. 读取测试 — 批量读取各电机位置/速度/负载
  4. 写入测试 — 安全地发送目标位置
  5. 连续读取 — 循环读取检测通信稳定性

使用方法:
  python tests/test_servo.py --port /dev/ttyACM0
  python tests/test_servo.py --port /dev/ttyACM0 --scan-only
  python tests/test_servo.py --port /dev/ttyACM0 --write-test --motor shoulder_pan --target 30.0
"""

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_connection(port: str, baudrate: int = 1000000) -> bool:
    """Test 1: 串口连接测试"""
    logger.info("=" * 60)
    logger.info("TEST 1: 连接测试")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "test_motor": Motor(1, "sts3215", MotorNormMode.DEGREES),
        },
    )

    try:
        bus.connect()
        logger.info(f"✓ 连接成功: {port}")
        bus.disconnect()
        return True
    except Exception as e:
        logger.error(f"✗ 连接失败: {e}")
        return False


def test_scan(port: str) -> dict:
    """Test 2: 扫描总线上所有电机"""
    logger.info("=" * 60)
    logger.info("TEST 2: Ping扫描测试")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    # 使用空电机列表来扫描
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "placeholder": Motor(1, "sts3215", MotorNormMode.DEGREES),
        },
    )

    try:
        bus.connect()
        # 扫描各个ID
        found_motors = {}
        for motor_id in range(1, 20):
            try:
                model = bus.read("Model_Number", f"placeholder", motor_id=motor_id)
                found_motors[motor_id] = model
                logger.info(f"  ✓ 电机 ID={motor_id}: model={model}")
            except Exception:
                pass

        if not found_motors:
            logger.warning("未找到任何电机")
        else:
            logger.info(f"共找到 {len(found_motors)} 个电机")

        bus.disconnect()
        return found_motors
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        return {}


def test_read(port: str, num_reads: int = 100) -> bool:
    """Test 3: 批量读取测试 — 验证sync_read的稳定性"""
    logger.info("=" * 60)
    logger.info("TEST 3: 批量读取测试")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
    )

    try:
        bus.connect()
        bus.disable_torque()

        # 读取当前位置
        logger.info("读取当前位置 (Present_Position):")
        pos = bus.sync_read("Present_Position")
        for name, val in pos.items():
            logger.info(f"  {name}: {val:.2f}°")

        # 连续读取测试
        logger.info(f"\n连续读取 {num_reads} 次测试:")
        errors = 0
        latencies = []

        for i in range(num_reads):
            start = time.perf_counter()
            try:
                pos = bus.sync_read("Present_Position")
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"  读取出错 #{errors}: {e}")
            dt_ms = (time.perf_counter() - start) * 1e3
            latencies.append(dt_ms)

        avg_ms = sum(latencies) / len(latencies)
        max_ms = max(latencies)
        min_ms = min(latencies)
        logger.info(f"  结果: {num_reads}次读取, {errors}次错误")
        logger.info(f"  延迟: avg={avg_ms:.1f}ms, min={min_ms:.1f}ms, max={max_ms:.1f}ms")
        logger.info(f"  频率: {1000.0/avg_ms:.0f} Hz")

        bus.disconnect()
        return errors < num_reads * 0.1  # 允许10%错误率

    except Exception as e:
        logger.error(f"读取测试失败: {e}")
        return False


def test_single_read(port: str) -> None:
    """Test 3b: 单电机逐个读取测试"""
    logger.info("=" * 60)
    logger.info("TEST 3b: 单电机逐个读取")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect()
        bus.disable_torque()

        for name, motor in motors.items():
            try:
                pos = bus.read("Present_Position", name)
                speed = bus.read("Present_Speed", name)
                load = bus.read("Present_Load", name)
                temp = bus.read("Present_Temperature", name)
                logger.info(
                    f"  {name} (ID={motor.id}): pos={pos:.2f}°, "
                    f"speed={speed:.1f}, load={load:.1f}, temp={temp}°C"
                )
            except Exception as e:
                logger.warning(f"  {name}: 读取失败 - {e}")

        bus.disconnect()
    except Exception as e:
        logger.error(f"单电机读取测试失败: {e}")


def test_write(port: str, motor_name: str, target_deg: float) -> bool:
    """Test 4: 写入测试 — 安全地发送目标位置"""
    logger.info("=" * 60)
    logger.info(f"TEST 4: 写入测试 (motor={motor_name}, target={target_deg}°)")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
    from lerobot.motors import Motor, MotorNormMode

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    if motor_name not in motors:
        logger.error(f"未知电机名: {motor_name}. 可选: {list(motors.keys())}")
        return False

    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect()

        # 先读取当前位置
        current_pos = bus.read("Present_Position", motor_name)
        logger.info(f"当前位置: {current_pos:.2f}°")
        logger.info(f"目标位置: {target_deg:.2f}°")

        # 安全检查: 限制最大角度变化
        delta = abs(target_deg - current_pos)
        if delta > 90:
            logger.warning(f"角度变化过大 ({delta:.1f}° > 90°)，请确认!")
            confirm = input("继续? (y/N): ")
            if confirm.lower() != "y":
                logger.info("已取消")
                bus.disconnect()
                return True

        # 启用扭矩并发送目标
        bus.enable_torque()
        bus.write("Goal_Position", motor_name, target_deg)
        logger.info("目标已发送, 等待2秒...")

        time.sleep(2)

        # 读取实际位置
        actual_pos = bus.read("Present_Position", motor_name)
        logger.info(f"实际位置: {actual_pos:.2f}°")

        # 回到原位置
        bus.write("Goal_Position", motor_name, current_pos)
        logger.info(f"回到原位置: {current_pos:.2f}°")
        time.sleep(1)

        bus.disconnect()
        return True

    except Exception as e:
        logger.error(f"写入测试失败: {e}")
        try:
            bus.disable_torque()
            bus.disconnect()
        except Exception:
            pass
        return False


def test_continuous_read(port: str, duration_s: float = 10.0, fps: int = 100) -> None:
    """Test 5: 连续读取 — 模拟实际使用场景的高频读取"""
    logger.info("=" * 60)
    logger.info(f"TEST 5: 连续读取测试 ({duration_s}s @ {fps}Hz)")
    logger.info("=" * 60)

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
    )

    try:
        bus.connect()
        bus.disable_torque()

        frame_count = 0
        error_count = 0
        start_time = time.perf_counter()
        latencies = []

        while time.perf_counter() - start_time < duration_s:
            loop_start = time.perf_counter()
            try:
                pos = bus.sync_read("Present_Position")
                frame_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    logger.warning(f"  读取出错: {e}")

            dt_ms = (time.perf_counter() - loop_start) * 1e3
            latencies.append(dt_ms)

            # FPS控制
            sleep_time = max(1.0 / fps - dt_ms / 1e3, 0)
            time.sleep(sleep_time)

        elapsed = time.perf_counter() - start_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        logger.info(f"  结果: {frame_count}帧, {error_count}次错误, {elapsed:.1f}s")
        logger.info(f"  实际频率: {frame_count/elapsed:.1f} Hz")
        logger.info(f"  平均读取延迟: {avg_latency:.1f}ms")

        # 打印最后一帧
        if frame_count > 0:
            pos = bus.sync_read("Present_Position")
            logger.info("  最后位置:")
            for name, val in pos.items():
                logger.info(f"    {name}: {val:.2f}°")

        bus.disconnect()

    except Exception as e:
        logger.error(f"连续读取测试失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="舵机/电机测试脚本")
    parser.add_argument("--port", type=str, required=True, help="串口设备路径 (e.g. /dev/ttyACM0)")
    parser.add_argument("--scan-only", action="store_true", help="仅扫描电机")
    parser.add_argument("--read-test", action="store_true", help="运行读取测试")
    parser.add_argument("--single-read", action="store_true", help="单电机逐个读取")
    parser.add_argument("--write-test", action="store_true", help="运行写入测试")
    parser.add_argument("--continuous", action="store_true", help="连续读取测试")
    parser.add_argument("--motor", type=str, default="shoulder_pan", help="写入测试的电机名")
    parser.add_argument("--target", type=float, default=0.0, help="写入测试的目标角度(度)")
    parser.add_argument("--duration", type=float, default=10.0, help="连续读取持续时间(秒)")
    parser.add_argument("--fps", type=int, default=100, help="连续读取目标频率")

    args = parser.parse_args()

    results = {}

    # Test 1: 连接测试
    results["connection"] = test_connection(args.port)

    if not results["connection"]:
        logger.error("连接测试失败，终止测试")
        sys.exit(1)

    # Test 2: 扫描测试
    if args.scan_only:
        test_scan(args.port)
        sys.exit(0)

    results["scan"] = bool(test_scan(args.port))

    # Test 3: 读取测试
    if args.read_test:
        results["read"] = test_read(args.port)

    # Test 3b: 单电机读取
    if args.single_read:
        test_single_read(args.port)

    # Test 4: 写入测试
    if args.write_test:
        results["write"] = test_write(args.port, args.motor, args.target)

    # Test 5: 连续读取
    if args.continuous:
        test_continuous_read(args.port, args.duration, args.fps)

    # 如果没指定任何测试，运行默认测试集
    if not any([args.scan_only, args.read_test, args.single_read, args.write_test, args.continuous]):
        logger.info("\n运行默认测试集 (连接 + 扫描 + 读取 + 单电机读取)")
        results["read"] = test_read(args.port, num_reads=50)
        test_single_read(args.port)

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")


if __name__ == "__main__":
    main()
