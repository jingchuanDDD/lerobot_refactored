#!/usr/bin/env python3
"""Quick FK/IK test for SO101."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lerobot.model.kinematics import RobotKinematics
from lerobot.rebuilt.arm_controller.so101_controller import SO101_MOTOR_NAMES
import numpy as np
from lerobot.utils.rotation import Rotation

# Load URDF (stripped version without mesh dependencies)
urdf = "robot_data/SO101/so101_kinematics.urdf"
kin = RobotKinematics(urdf)

print("URDF joint names:", kin.joint_names)
print("SO101_MOTOR_NAMES:", SO101_MOTOR_NAMES)
print("Match:", kin.joint_names == SO101_MOTOR_NAMES)
print()

# Test 1: FK with zero positions (6 joints including gripper)
test_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
T = kin.forward_kinematics(test_q)
pos = T[:3, 3]
quat = Rotation.from_matrix(T[:3, :3]).as_quat()
print("=== Test 1: FK (all zeros) ===")
print(f"Pos: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}")
print(f"Quat: qw={quat[3]:.6f} qx={quat[0]:.6f} qy={quat[1]:.6f} qz={quat[2]:.6f}")
print()

# Test 2: FK with non-zero positions
test_q2 = np.array([30.0, -20.0, 45.0, 10.0, 90.0, 0.0], dtype=float)
T2 = kin.forward_kinematics(test_q2)
pos2 = T2[:3, 3]
quat2 = Rotation.from_matrix(T2[:3, :3]).as_quat()
print("=== Test 2: FK (nonzero joints) ===")
for name, val in zip(SO101_MOTOR_NAMES, test_q2):
    print(f"  {name}: {val:.1f}°")
print(f"Pos: x={pos2[0]:.4f} y={pos2[1]:.4f} z={pos2[2]:.4f}")
print(f"Quat: qw={quat2[3]:.6f} qx={quat2[0]:.6f} qy={quat2[1]:.6f} qz={quat2[2]:.6f}")
print()

# Test 3: IK Round Trip (input → FK → IK → same input)
print("=== Test 3: IK Round Trip ===")
q_sol = kin.inverse_kinematics(test_q2, T2)
for name, inp, out in zip(SO101_MOTOR_NAMES, test_q2, q_sol):
    print(f"  {name}: in={inp:.3f}° out={out:.3f}° err={abs(inp-out):.4f}°")
err = np.max(np.abs(test_q2 - q_sol))
print(f"Max joint error: {err:.6f}°")
print(f"IK Round Trip: {'PASS' if err < 1e-4 else 'FAIL (tolerance 1e-4°)'}")
print()

# Test 4: IK from different initial guess
print("=== Test 4: IK from different init ===")
init_q = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
q_sol2 = kin.inverse_kinematics(init_q, T2)
T_check = kin.forward_kinematics(q_sol2)
pos_check = T_check[:3, 3]
err_pos = np.linalg.norm(pos2 - pos_check)
print(f"Init guess: {dict(zip(SO101_MOTOR_NAMES[:3], init_q[:3]))} ...")
print(f"Target pos: x={pos2[0]:.4f} y={pos2[1]:.4f} z={pos2[2]:.4f}")
print(f"Got pos:    x={pos_check[0]:.4f} y={pos_check[1]:.4f} z={pos_check[2]:.4f}")
print(f"Position err: {err_pos:.6f}m")
print(f"IK from diff init: {'PASS' if err_pos < 0.001 else 'WARN'}")
print()

# Test 5: FK->IK cycle with non-trivial gripper
print("=== Test 5: FK->IK with gripper ===")
q_with_grip = np.array([10.0, 15.0, -30.0, -20.0, 45.0, 50.0], dtype=float)
T3 = kin.forward_kinematics(q_with_grip)
q_back = kin.inverse_kinematics(np.zeros(6), T3)
err2 = np.max(np.abs(q_with_grip[:5] - q_back[:5]))
print(f"Max body joint error (excl grip): {err2:.6f}°")
print(f"FK->IK cycle: {'PASS' if err2 < 1e-4 else 'FAIL'}")

print()
print("=" * 50)
print("ALL FK/IK TESTS COMPLETE")
