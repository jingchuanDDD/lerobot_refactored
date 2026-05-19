"""Lerobot Sim — MuJoCo-based simulation backend for SO101 arm.

Modules:
  - SimEnv: MuJoCo environment wrapper (model/data/rendering).
  - SimArmController: ArmController implementation in simulation.
  - SimPerception: PerceptionModule backed by MuJoCo rendering.
"""

from .env import SimEnv
from .sim_arm_controller import SimArmController
from .sim_perception import SimPerception

__all__ = ["SimEnv", "SimArmController", "SimPerception"]
