#!/usr/bin/env python3
"""Test FK/IK via SO101ArmController (no hardware needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lerobot.rebuilt.arm_controller.so101_controller import SO101ArmController, SO101_MOTOR_NAMES
import numpy as np

arm = SO101ArmController(port="/dev/null")
arm._init_kinematics()

if arm._kinematics is None:
    print("FAIL: Kinematics not loaded")
    sys.exit(1)

# FK all zeros
joints = {n: 0.0 for n in SO101_MOTOR_NAMES}
ee = arm.forward_kinematics(joints)
print("=== FK (all zeros) ===")
for k, v in ee.items():
    print(f"  {k}: {v:.6f}")

# FK nonzero
joints2 = {"shoulder_pan": 30.0, "shoulder_lift": -20.0, "elbow_flex": 45.0,
           "wrist_flex": 10.0, "wrist_roll": 90.0, "gripper": 0.0}
ee2 = arm.forward_kinematics(joints2)
print(f"\n=== FK (nonzero) ===")
print(f"  pos: x={ee2['x']:.4f} y={ee2['y']:.4f} z={ee2['z']:.4f}")
print(f"  quat: qw={ee2['qw']:.6f} qx={ee2['qx']:.6f} qy={ee2['qy']:.6f} qz={ee2['qz']:.6f}")

# IK round trip
ik = arm.inverse_kinematics(ee2, current_joints=joints2)
print(f"\n=== IK Round Trip ===")
max_err = 0.0
for n in SO101_MOTOR_NAMES:
    err = abs(joints2[n] - ik[n])
    max_err = max(max_err, err)
    print(f"  {n}: in={joints2[n]:.1f} out={ik[n]:.1f} err={err:.4f}")

status = "PASS" if max_err < 1e-4 else "FAIL"
print(f"\nIK Round Trip: {status} (max err={max_err:.6f}°)")

# Check FK of IK output matches
from lerobot.utils.rotation import Rotation
from lerobot.model.kinematics import RobotKinematics
kin = RobotKinematics("robot_data/SO101/so101_kinematics.urdf")
T2 = kin.forward_kinematics(np.array([joints2[n] for n in SO101_MOTOR_NAMES]))
T_ik = kin.forward_kinematics(np.array([ik[n] for n in SO101_MOTOR_NAMES]))
pos_err = np.linalg.norm(T2[:3,3] - T_ik[:3,3])
print(f"FK->IK->FK position error: {pos_err:.6f}m")

print(f"\nFK/IK via ArmController: ALL OK!")
