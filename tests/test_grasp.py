#!/usr/bin/env python3
"""Grasping test: close gripper around cube with physics enabled.

Usage:
    python tests/test_grasp.py          # headless (prints diagnostics)
    python tests/test_grasp.py --gui    # with MuJoCo viewer
"""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np

# Allow running from project root or tests/ dir
sys.path.insert(0, "/home/data/xhw/xhw/lerobot_fix")
sys.path.insert(0, "/home/data/xhw/xhw/lerobot_fix/src")

from lerobot.rebuilt.sim.env import SimEnv


# Pre-computed pose that places the open gripper around the cube
# (cube at pos="0.018 0.119 0.621")
PREGRASP = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.0,    # arm reaches forward-down
    "elbow_flex":   80.0,      # elbow bent to position EE
    "wrist_flex":   -55.0,     # wrist compensates
    "wrist_roll":   0.0,
    "gripper":      100.0,     # fully open
}

# Same pose but gripper closed
CLOSED = dict(PREGRASP)
CLOSED["gripper"] = 0.0


def print_contact_info(env: SimEnv, label: str = ""):
    """Print current contact pairs and cube position."""
    ncon = env.data.ncon
    cube_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube") if 'mujoco' in dir() else -1
    # Get cube position from its freejoint qpos
    cube_qpos = None
    for i in range(env.model.njnt):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name == "free":
            addr = env.model.jnt_qposadr[i]
            cube_qpos = env.data.qpos[addr:addr+7].copy()
            break

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Contacts: {ncon}")
    for c in range(ncon):
        contact = env.data.contact[c]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) if 'mujoco' in dir() else str(contact.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) if 'mujoco' in dir() else str(contact.geom2)
        dist = contact.dist
        force = np.linalg.norm(contact.force) if hasattr(contact, 'force') and contact.force is not None else 0
        print(f"    {g1} <-> {g2}:  dist={dist:.4f}  |force|={force:.4f}")
    if cube_qpos is not None:
        print(f"  Cube pos: x={cube_qpos[0]:.4f} y={cube_qpos[1]:.4f} z={cube_qpos[2]:.4f}")
    print(f"{'='*50}")


def run_headless_test():
    """Run grasp sequence without viewer, print diagnostics."""
    import mujoco as mj  # import here so --gui also works

    env = SimEnv()
    print(f"SimEnv loaded. Substeps per frame: {env.substeps}, dt={env.model.opt.timestep}")

    # --- Phase 1: Move to pregrasp with direct control ---
    print("\n[Phase 1] Moving to pregrasp (direct control)...")
    for i in range(30):
        t = min(i / 29.0, 1.0)
        action = {k: v * t for k, v in PREGRASP.items()}
        env.step(action, physics=False)

    state = env.get_state()
    ee = env.get_ee_pos()
    print(f"  Pregrasp reached. EE pos: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    print(f"  Gripper: {state['gripper']:.1f}")
    print_contact_info(env, "Before closing")

    # Record cube position before closing
    cube_pos_before = _get_cube_pos(env)

    # --- Phase 2: Close gripper with PHYSICS ---
    print("\n[Phase 2] Closing gripper with physics (mj_step)...")
    gripper_open = PREGRASP["gripper"]
    gripper_closed = CLOSED["gripper"]
    n_close_frames = 60

    for i in range(n_close_frames):
        t = min(i / (n_close_frames - 1), 1.0)
        # Smooth interpolation
        t_smooth = t * t * (3 - 2 * t)  # smoothstep
        action = dict(PREGRASP)
        action["gripper"] = gripper_open + (gripper_closed - gripper_open) * t_smooth
        env.step(action, physics=True)

        if i % 15 == 0 or i == n_close_frames - 1:
            print_contact_info(env, f"Close frame {i}/{n_close_frames-1}")

    cube_pos_after = _get_cube_pos(env)
    ee_final = env.get_ee_pos()

    print(f"\n{'='*50}")
    print("  RESULTS")
    print(f"{'='*50}")
    print(f"  Cube moved:  ({cube_pos_before[0]-cube_pos_after[0]:+.4f}, "
          f"{cube_pos_before[1]-cube_pos_after[1]:+.4f}, "
          f"{cube_pos_before[2]-cube_pos_after[2]:+.4f})")
    displacement = np.linalg.norm(cube_pos_before - cube_pos_after)
    print(f"  Total displacement: {displacement:.4f} m")
    print(f"  Final EE pos: [{ee_final[0]:.3f}, {ee_final[1]:.3f}, {ee_final[2]:.3f}]")
    final_grip = env.get_state()["gripper"]
    print(f"  Final gripper: {final_grip:.1f}")

    if displacement > 0.005:
        print("\n  *** SUCCESS: Cube was affected by gripper! ***")
    else:
        print("\n  *** FAILURE: Cube did not move — check collision setup ***")

    env.close()


def run_gui_test():
    """Run grasp sequence with interactive viewer."""
    import mujoco as mj

    env = SimEnv()

    # Move to pregrasp first
    print("Moving to pregrasp...")
    for i in range(30):
        t = min(i / 29.0, 1.0)
        action = {k: v * t for k, v in PREGRASP.items()}
        env.step(action, physics=False)

    print("Launching viewer... Close the window to exit.")
    viewer = mj.MjViewer(env.model, env.data)

    frame = [0]
    phase = ["closing"]  # mutable container
    start_time = [time.perf_counter()]

    def key_callback(code):
        if code == mj.mjtEvent.mjEVENT_KEY and viewer.keydata == ord('r'):
            # Reset
            env.reset()
            phase[0] = "closing"
            frame[0] = 0
            start_time[0] = time.perf_counter()
            print("[r] Reset")
        elif code == mj.mjtEvent.mjEVENT_KEY and viewer.keydata == ord('o'):
            phase[0] = "opening"
            frame[0] = 0
            print("[o] Opening...")
        elif code == mj.mjtEvent.mjEVENT_KEY and viewer.keydata == ord('c'):
            phase[0] = "closing"
            frame[0] = 0
            print("[c] Closing...")

    while not viewer.is_running:
        # Determine gripper target based on phase
        elapsed = time.perf_counter() - start_time[0]
        t_clip = min(elapsed / 2.0, 1.0)  # 2 second transition
        t_smooth = t_clip * t_clip * (3 - 2 * t_clip)

        if phase[0] == "closing":
            grip_val = 100.0 * (1 - t_smooth)
        else:
            grip_val = 100.0 * t_smooth

        grip_val = max(0, min(100, grip_val))

        action = {
            "shoulder_pan": PREGRASP["shoulder_pan"],
            "shoulder_lift": PREGRASP["shoulder_lift"],
            "elbow_flex": PREGRASP["elbow_flex"],
            "wrist_flex": PREGRASP["wrist_flex"],
            "wrist_roll": PREGRASP["wrist_roll"],
            "gripper": grip_val,
        }
        env.step(action, physics=True)

        # Update viewer
        viewer.sync()
        viewer.key_callback = key_callback

        frame[0] += 1
        if frame[0] % 30 == 0:
            cube_p = _get_cube_pos(env)
            ee_p = env.get_ee_pos()
            print(f"  frame={frame[0]}  grip={grip_val:.0f}%  "
                  f"cube=({cube_p[0]:.3f},{cube_p[1]:.3f},{cube_p[2]:.3f})  "
                  f"ncon={env.data.ncon}")

    env.close()


def _get_cube_pos(env) -> np.ndarray:
    """Extract cube body position from freejoint qpos."""
    import mujoco as mj
    for i in range(env.model.njnt):
        name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_JOINT, i)
        if name == "free":
            addr = env.model.jnt_qposadr[i]
            return env.data.qpos[addr:addr+3].copy()
    return np.zeros(3)


def main():
    parser = argparse.ArgumentParser(description="Grasping test with physics")
    parser.add_argument("--gui", action="store_true", help="Launch MuJoCo viewer")
    args = parser.parse_args()

    import mujoco  # ensure available

    if args.gui:
        run_gui_test()
    else:
        run_headless_test()


if __name__ == "__main__":
    main()
