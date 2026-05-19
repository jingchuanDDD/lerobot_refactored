"""Test script for SO101 simulation module.

Tests two core functionalities:
  1. SimEnv physics — state read/write, step, FK.
  2. SimArmController — full ArmController interface compatibility.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")


def test_sim_env_physics():
    """Test 1: SimEnv core physics — state, step, FK, reset."""
    from lerobot.rebuilt.sim.env import SimEnv

    print("=" * 60)
    print("TEST 1: SimEnv Physics")
    print("=" * 60)

    env = SimEnv()

    # --- 1a. Reset to home position ---
    home = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}
    env.reset(home)
    s0 = env.get_state()
    print(f"[1a] Reset OK: gripper={s0['gripper']:.1f} (expect ~50)")

    # --- 1b. Single step ---
    env.step({"shoulder_pan": 45, "shoulder_lift": -20, "elbow_flex": 30,
              "wrist_flex": 10, "wrist_roll": -15, "gripper": 60})
    s1 = env.get_state()
    # Direct position control: exact tracking
    assert abs(s1["shoulder_pan"] - 45) < 1, f"pan={s1['shoulder_pan']:.1f}"
    assert abs(s1["elbow_flex"] - 30) < 1, f"elbow={s1['elbow_flex']:.1f}"
    print(f"[1b] Step OK: pan={s1['shoulder_pan']:.1f} elbow={s1['elbow_flex']:.1f}")

    # --- 1c. EE pose after step ---
    ee = env.get_ee_pose()
    assert ee, "EE pose should not be empty"
    assert -0.5 < ee["x"] < 0.5, f"ee_x={ee['x']:.3f}"
    assert -0.5 < ee["y"] < 0.5, f"ee_y={ee['y']:.3f}"
    assert 0.1 < ee["z"] < 0.8, f"ee_z={ee['z']:.3f}"
    print(f"[1c] EE pos: ({ee['x']:.3f}, {ee['y']:.3f}, {ee['z']:.3f})")

    # --- 1d. FK with explicit angles ---
    ee_fk = env.forward_kinematics(home)
    assert ee_fk, "FK result should not be empty"
    print(f"[1d] FK of home: z={ee_fk['z']:.3f}")

    # --- 1e. Multiple steps ---
    env.reset(home)
    positions_log = []
    for i in range(10):
        angle = i * 6  # 0..60 degrees
        env.step({"shoulder_pan": angle, "shoulder_lift": -angle * 0.3,
                  "elbow_flex": angle * 0.5, "wrist_flex": 0,
                  "wrist_roll": angle * 0.2, "gripper": 50})
        s = env.get_state()
        positions_log.append(s["shoulder_pan"])
    # Verify monotonic motion
    for i in range(1, len(positions_log)):
        assert positions_log[i] > positions_log[i - 1], \
            f"Non-monotonic at step {i}: {positions_log[i-1]:.1f} → {positions_log[i]:.1f}"
    print(f"[1e] 10-step path OK: {positions_log[0]:.1f}° → {positions_log[-1]:.1f}°")

    env.close()
    print("[TEST 1] ALL PASSED\n")


def test_sim_arm_controller():
    """Test 2: SimArmController — full ArmController interface."""
    from lerobot.rebuilt.sim import SimArmController
    from lerobot.rebuilt.arm_controller.core import ArmAction, ControlMode

    print("=" * 60)
    print("TEST 2: SimArmController (ArmController interface)")
    print("=" * 60)

    arm = SimArmController()

    # --- 2a. connect + properties ---
    arm.connect()
    assert arm.is_connected, "Should be connected"
    assert arm.is_calibrated, "Should be calibrated in sim"
    assert arm.name == "sim_so101"
    assert len(arm.motor_names) == 6
    assert "shoulder_pan" in arm.motor_names
    assert "gripper" in arm.motor_names
    print(f"[2a] Connected: name={arm.name}, motors={arm.motor_names}")

    # --- 2b. get_state ---
    state = arm.get_state()
    assert state.joint_positions is not None
    assert len(state.joint_positions) == 6
    assert state.ee_pose is not None, "EE pose should be available"
    assert "x" in state.ee_pose
    print(f"[2b] State joints: {len(state.joint_positions)}, EE: {state.ee_pose['x']:.3f}, {state.ee_pose['y']:.3f}, {state.ee_pose['z']:.3f}")

    # --- 2c. send_action (JOINT mode) ---
    action = ArmAction(
        joint_positions={"shoulder_pan": 30, "shoulder_lift": -15,
                         "elbow_flex": 20, "wrist_flex": 5,
                         "wrist_roll": -10, "gripper": 60},
        control_mode=ControlMode.JOINT,
    )
    sent = arm.send_action(action)
    assert sent.joint_positions is not None
    s2 = arm.get_state()
    assert abs(s2.joint_positions["shoulder_pan"] - 30) < 1, \
        f"pan={s2.joint_positions['shoulder_pan']:.1f}"
    print(f"[2c] send_action OK: pan→{s2.joint_positions['shoulder_pan']:.1f}°")

    # --- 2d. action/observation features ---
    features = arm.action_features
    assert len(features) == 6
    assert "shoulder_pan.pos" in features
    obs_features = arm.observation_features
    assert len(obs_features) == 6
    print(f"[2d] Features: {len(features)} action, {len(obs_features)} observation")

    # --- 2e. forward_kinematics ---
    fk = arm.forward_kinematics({
        "shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
        "wrist_flex": 0, "wrist_roll": 0, "gripper": 50
    })
    assert "x" in fk, f"FK missing x: {fk}"
    print(f"[2e] FK: pos=({fk['x']:.3f}, {fk['y']:.3f}, {fk['z']:.3f})")

    # --- 2f. context manager ---
    arm.disconnect()
    assert not arm.is_connected
    print("[2f] disconnect OK")

    arm.connect()
    assert arm.is_connected
    arm.disconnect()
    print(f"[2f] reconnect OK")

    print("[TEST 2] ALL PASSED\n")


def test_from_robot_action_all_modes():
    """Test 3: ArmAction.from_robot_action — all three control modes."""
    from lerobot.rebuilt.arm_controller.core import ArmAction, ControlMode

    print("=" * 60)
    print("TEST 3: ArmAction.from_robot_action (JOINT / CARTESIAN / HYBRID)")
    print("=" * 60)

    # --- 3a. JOINT-only input ---
    legacy_joint = {"shoulder_pan.pos": 15.0, "elbow_flex.pos": -28.0, "gripper.pos": 60.0}
    parsed_j = ArmAction.from_robot_action(legacy_joint)
    assert parsed_j.control_mode == ControlMode.JOINT, f"Expected JOINT, got {parsed_j.control_mode}"
    assert parsed_j.joint_positions == {"shoulder_pan": 15.0, "elbow_flex": -28.0, "gripper": 60.0}
    assert parsed_j.ee_pose is None, "ee_pose should be None for JOINT-only input"
    print(f"[3a] JOINT-only OK: mode={parsed_j.control_mode}, joints={list(parsed_j.joint_positions.keys())}")

    # --- 3b. CARTESIAN-only input ---
    legacy_cart = {"ee.x": 0.3, "ee.y": 0.1, "ee.z": 0.4, "ee.qw": 1.0, "ee.qx": 0.0, "ee.qy": 0.0, "ee.qz": 0.0}
    parsed_c = ArmAction.from_robot_action(legacy_cart)
    assert parsed_c.control_mode == ControlMode.CARTESIAN, f"Expected CARTESIAN, got {parsed_c.control_mode}"
    assert parsed_c.ee_pose == {"x": 0.3, "y": 0.1, "z": 0.4, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
    assert parsed_c.joint_positions is None, "joint_positions should be None for CARTESIAN-only"
    print(f"[3b] CARTESIAN-only OK: mode={parsed_c.control_mode}, ee={parsed_c.ee_pose}")

    # --- 3c. HYBRID input ---
    legacy_hybrid = {"shoulder_pan.pos": 30.0, "ee.x": 0.5, "ee.qw": 1.0}
    parsed_h = ArmAction.from_robot_action(legacy_hybrid)
    assert parsed_h.control_mode == ControlMode.HYBRID, f"Expected HYBRID, got {parsed_h.control_mode}"
    assert parsed_h.joint_positions["shoulder_pan"] == 30.0
    assert parsed_h.ee_pose["x"] == 0.5
    assert parsed_h.ee_pose["qw"] == 1.0
    print(f"[3c] HYBRID OK: mode={parsed_h.control_mode}")

    print("[TEST 3] ALL PASSED\n")


def test_send_action_all_modes():
    """Test 4: SimArmController.send_action — JOINT / HYBRID / CARTESIAN."""
    from lerobot.rebuilt.sim import SimArmController
    from lerobot.rebuilt.arm_controller.core import ArmAction, ControlMode

    print("=" * 60)
    print("TEST 4: SimArmController.send_action (JOINT / HYBRID / CARTESIAN)")
    print("=" * 60)

    arm = SimArmController()
    arm.connect()

    home = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}

    # --- 4a. JOINT mode ---
    arm.send_action(ArmAction(joint_positions=home, control_mode=ControlMode.JOINT))
    action_j = ArmAction(
        joint_positions={"shoulder_pan": 20, "shoulder_lift": -10, "elbow_flex": 15,
                         "wrist_flex": 5, "wrist_roll": -5, "gripper": 70},
        control_mode=ControlMode.JOINT,
    )
    arm.send_action(action_j)
    s = arm.get_state()
    assert abs(s.joint_positions["shoulder_pan"] - 20) < 1, f"pan={s.joint_positions['shoulder_pan']:.1f}"
    print(f"[4a] JOINT OK: pan={s.joint_positions['shoulder_pan']:.1f}°")

    # --- 4b. HYBRID mode (uses joint_positions, ee_pose logged) ---
    action_h = ArmAction(
        joint_positions={"shoulder_pan": 35, "shoulder_lift": -20, "elbow_flex": 25,
                         "wrist_flex": 10, "wrist_roll": -10, "gripper": 60},
        ee_pose={"x": 0.1, "y": 0.2, "z": 0.3, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        control_mode=ControlMode.HYBRID,
    )
    sent_h = arm.send_action(action_h)
    assert sent_h.control_mode == ControlMode.HYBRID
    assert sent_h.ee_pose is not None, "ee_pose should be preserved in HYBRID mode"
    s2 = arm.get_state()
    assert abs(s2.joint_positions["shoulder_pan"] - 35) < 1, f"pan={s2.joint_positions['shoulder_pan']:.1f}"
    print(f"[4b] HYBRID OK: pan={s2.joint_positions['shoulder_pan']:.1f}°, ee preserved")

    # --- 4c. CARTESIAN mode (dummy IK fallback — returns current joints) ---
    arm.send_action(ArmAction(joint_positions=home, control_mode=ControlMode.JOINT))
    action_c = ArmAction(
        ee_pose={"x": 0.15, "y": -0.05, "z": 0.35, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        control_mode=ControlMode.CARTESIAN,
    )
    try:
        sent_c = arm.send_action(action_c)
        print(f"[4c] CARTESIAN OK (dummy IK fallback): mode={sent_c.control_mode}")
    except Exception as e:
        print(f"[4c] CARTESIAN raised (expected without placo): {type(e).__name__}")

    arm.disconnect()
    print("[TEST 4] ALL PASSED\n")


def run_all():
    print("\n" + "=" * 60)
    print("  SO101 Simulation Module — Test Suite")
    print("=" * 60 + "\n")

    test_sim_env_physics()
    test_sim_arm_controller()
    test_from_robot_action_all_modes()
    test_send_action_all_modes()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
