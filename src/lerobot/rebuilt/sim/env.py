"""SimEnv — lightweight MuJoCo environment wrapper for SO101 arm simulation.

Handles model/data lifecycle, stepping, rendering, and degree↔radian conversion.
Rendering is lazily initialized — physics-only mode works headless.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco
from mujoco import mjtObj

logger = logging.getLogger(__name__)

# Joint names matching SO101 motor names (URDF order)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_DEFAULT_MJCF = Path(__file__).parent / "configs" / "so101_sim.xml"


@dataclass
class SimOptions:
    """Options to pass into SimEnv.__init__."""
    fps: int = 30
    scene_width: int = 640
    scene_height: int = 480
    physics_timestep: float = 0.002


class SimEnv:
    """SO101 simulation environment wrapping a MuJoCo model + data.

    Joint positions are stored internally in RADIANS, but the public API
    (get_state, set_state, step) works in DEGREES for consistency with real
    hardware.  The gripper uses 0‑100 range (matching RANGE_0_100 norm mode).

    Rendering is *lazy*: physics-only usage works head‑less.  When a render
    method is called and no OpenGL context is available a clear error is raised
    describing which of (x11 / EGL / OSMesa) is missing.
    """

    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        fps: int | None = None,
        scene_width: int | None = None,
        scene_height: int | None = None,
        physics_timestep: float | None = None,
        joint_offsets: dict[str, float] | None = None,
    ):
        """Args:
            joint_offsets: degrees to add per joint when converting motor→radians.
                e.g. {"wrist_roll": 90} compensates for URDF reference vs real zero.
        """
        mjcf_path = str(mjcf_path or _DEFAULT_MJCF)
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)
        self._renderer = None  # lazy init

        # Joint offsets in degrees (URDF reference → real motor zero)
        self._offsets: dict[str, float] = dict(joint_offsets or {})

        # Physics
        phys_ts = physics_timestep or SimOptions.physics_timestep
        self._model.opt.timestep = phys_ts
        actual_fps = fps or SimOptions.fps
        self._frame_dt = 1.0 / actual_fps
        self._substeps = max(1, int(self._frame_dt / phys_ts + 0.5))
        self._fps = actual_fps

        # Resolution for potential renderer
        self._width = scene_width or SimOptions.scene_width
        self._height = scene_height or SimOptions.scene_height

        # Joint id lookup
        self._joint_ids: dict[str, int] = {}
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mjtObj.mjOBJ_JOINT, i)
            if name in JOINT_NAMES:
                self._joint_ids[name] = i

        # Actuator id lookup
        self._joint_to_actuator: dict[str, int] = {}
        for i in range(self._model.nu):
            joint_name = mujoco.mj_id2name(
                self._model, mjtObj.mjOBJ_ACTUATOR, i
            )
            if joint_name:
                # Actuator name is e.g. "act_shoulder_pan"
                mot = joint_name.replace("act_", "")
                if mot in JOINT_NAMES:
                    self._joint_to_actuator[mot] = i

        if len(self._joint_ids) != len(JOINT_NAMES):
            missing = set(JOINT_NAMES) - set(self._joint_ids)
            raise RuntimeError(f"Missing joints in MJCF: {missing}")

        # EE site
        self._ee_site_id = -1
        try:
            self._ee_site_id = mujoco.mj_name2id(
                self._model, mjtObj.mjOBJ_SITE, "ee_site"
            )
        except Exception:
            logger.warning("No ee_site found in MJCF — FK will return {}")

        # Cameras
        self._camera_ids: dict[str, int] = {}
        for i in range(self._model.ncam):
            name = mujoco.mj_id2name(self._model, mjtObj.mjOBJ_CAMERA, i)
            self._camera_ids[name] = i

        logger.info(
            "SimEnv loaded: %d joints, %d bodies, %d substeps @ %d fps",
            self._model.njnt, self._model.nbody, self._substeps, actual_fps,
        )

    # -- properties -------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def renderer(self):
        return self._get_renderer()

    @property
    def joint_names(self) -> list[str]:
        return list(JOINT_NAMES)

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def substeps(self) -> int:
        return self._substeps

    # -- unit conversion -------------------------------------------------

    def _motor_to_radians(self, motor: dict[str, float]) -> np.ndarray:
        """Convert motor units (degrees / 0-100) → radians array."""
        q = self._data.qpos.copy()
        for name, val in motor.items():
            if name not in self._joint_ids:
                continue
            addr = self._model.jnt_qposadr[self._joint_ids[name]]
            offset_deg = self._offsets.get(name, 0.0)
            if name == "gripper":
                q[addr] = float(val) / 100.0 * 1.92033 - 0.175 + offset_deg * np.pi / 180.0
            else:
                q[addr] = (float(val) + offset_deg) * np.pi / 180.0
        return q

    def _radians_to_motor(self, q: np.ndarray | None = None) -> dict[str, float]:
        """Convert MuJoCo qpos (radians) → motor units (degrees / 0-100)."""
        if q is None:
            q = self._data.qpos
        result: dict[str, float] = {}
        for name in JOINT_NAMES:
            addr = self._model.jnt_qposadr[self._joint_ids[name]]
            rad = float(q[addr])
            offset_deg = self._offsets.get(name, 0.0)
            if name == "gripper":
                result[name] = (rad + 0.175) / 1.92033 * 100.0
            else:
                result[name] = rad * 180.0 / np.pi - offset_deg
        return result

    # -- core API --------------------------------------------------------

    def get_state(self) -> dict[str, float]:
        """Read joint positions in motor units (degrees / 0-100)."""
        return self._radians_to_motor()

    def set_state(self, positions: dict[str, float]):
        """Set joint positions without advancing physics."""
        q = self._motor_to_radians(positions)
        self._data.qpos[:] = q
        mujoco.mj_forward(self._model, self._data)

    def step(self, action: dict[str, float]):
        """Apply position action (motor units) and advance one frame.

        Direct position control: sets qpos to target and runs forward kinematics.
        No physics substeps — joints track targets exactly (teleop-in-sim mode).
        """
        q_target = self._motor_to_radians(action)
        self._data.qpos[:] = q_target
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def forward(self):
        """Run forward kinematics (no time advance)."""
        mujoco.mj_forward(self._model, self._data)

    def reset(self, positions: dict[str, float] | None = None):
        """Reset to a given joint configuration or to zero."""
        mujoco.mj_resetData(self._model, self._data)
        if positions:
            self.set_state(positions)
        else:
            mujoco.mj_forward(self._model, self._data)

    def get_ee_pose(self) -> dict[str, float]:
        """End-effector pose from ee_site (position + quaternion)."""
        if self._ee_site_id < 0:
            return {}
        pos = self._data.site_xpos[self._ee_site_id]
        xmat = self._data.site_xmat[self._ee_site_id].reshape(3, 3)
        quat = self._mat_to_quat(xmat)
        return {
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "qw": float(quat[3]), "qx": float(quat[0]),
            "qy": float(quat[1]), "qz": float(quat[2]),
        }

    def get_ee_pos(self) -> np.ndarray:
        """End-effector [x, y, z] as float64 array."""
        if self._ee_site_id < 0:
            return np.zeros(3)
        return self._data.site_xpos[self._ee_site_id].copy()

    # -- kinematics ------------------------------------------------------

    def forward_kinematics(
        self, joint_positions: dict[str, float]
    ) -> dict[str, float]:
        """Compute EE pose from joint positions (motor units)."""
        current = self._data.qpos.copy()
        try:
            self.set_state(joint_positions)
            return self.get_ee_pose()
        finally:
            self._data.qpos[:] = current
            mujoco.mj_forward(self._model, self._data)

    # -- rendering (lazy) -----------------------------------------------

    def _get_renderer(self):
        if self._renderer is None:
            self._init_renderer()
        return self._renderer

    def _init_renderer(self):
        try:
            # Try GLFW first (needs X11 display)
            self._renderer = mujoco.Renderer(
                self._model, self._height, self._width
            )
        except Exception as e:
            msg = (
                "Renderer initialization failed.\n"
                "  GLFW (default) requires X11 display — set DISPLAY=:0\n"
                "  or install xvfb:  sudo apt install xvfb\n"
                f"  Original error: {e}"
            )
            raise RuntimeError(msg) from e

    def render_rgb(self, cam_name: str | None = None) -> np.ndarray:
        """Render RGB image (HWC uint8)."""
        r = self._get_renderer()
        r.update_scene(self._data, camera=cam_name)
        return r.render().copy()

    def render_depth(self, cam_name: str | None = None) -> np.ndarray:
        """Render depth image (HxW float32)."""
        r = self._get_renderer()
        r.enable_depth_rendering()
        r.update_scene(self._data, camera=cam_name)
        depth = r.render()
        r.disable_depth_rendering()
        return depth.copy()

    # -- internal -------------------------------------------------------

    @staticmethod
    def _mat_to_quat(m: np.ndarray) -> list[float]:
        """Rotation matrix 3×3 → quaternion [qx, qy, qz, qw]."""
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            return [
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
                0.25 * s,
            ]
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            return [
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[2, 0] + m[0, 2]) / s,
                (m[2, 1] - m[1, 2]) / s,
            ]
        if m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            return [
                (m[1, 0] + m[0, 1]) / s,
                0.25 * s,
                (m[2, 1] + m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
            ]
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        return [
            (m[2, 0] + m[0, 2]) / s,
            (m[2, 1] + m[1, 2]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]

    def close(self):
        """Release renderer resources."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def __del__(self):
        if hasattr(self, '_renderer'):
            self.close()
