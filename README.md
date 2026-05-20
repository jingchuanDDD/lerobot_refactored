# LeRobot Refactored

This repository is a refactored version of HuggingFace's LeRobot project.
The goal is to modularize the original codebase and extend it with
teleoperation and simulation replay capabilities for robotic manipulation research.

This work is conducted as part of an internship task focusing on
robot learning system engineering.

---

## Overview

The original LeRobot codebase tightly couples robot control, data collection,
teleoperation, and perception logic.
To improve extensibility and support real-to-sim workflows,
this repository reorganizes the system into clear functional modules.

This repository is based on HuggingFace's LeRobot project.
The original codebase is preserved to maintain compatibility and reference.

New functionalities and refactored components are introduced in separate modules,
allowing incremental refactoring without breaking the original workflow.

In addition, new features are introduced to support:
- Real-robot teleoperation with synchronized simulation replay
- Data replay in simulation for policy debugging and evaluation

---

## Repository Structure

The repository currently contains two main parts:

- **Original LeRobot Codebase**
  - The original directory structure from HuggingFace LeRobot
  - Kept unchanged for compatibility and baseline reference

- **Refactored and Extended Modules**
  - Located in a separate directory: `src/lerobot/rebuilt`
  - Contains newly introduced modules for:
    - Robot arm control – `rebuilt/arm_controller`
    - Teleoperation – `rebuilt/teleop_module`
    - Data collection – `rebuilt/data_collector`
    - Perception – `rebuilt/perception_module`
    - Simulation teleoperation and replay – `rebuilt/sim`
    - Refactored main control loop for data collection – `rebuilt/unified_loop.py`

## Testing

Test scripts are located under `tests` and include:

- Module-level sanity checks and control loop validation - `test_lerobot_modules.py`
- Teleoperation behavior validation in simulation - `test_scenario_b.py`
- Data replay consistency validation - `test_data_replay.py`
---

## System Modularization

The system is refactored into the following core modules:

### 1. Robot Arm Control Module
- Low-level robot control interface
- Joint / Cartesian command abstraction
- Unified interface for real robot and simulation backends

### 2. Teleoperation Module
- Human-in-the-loop teleoperation for robotic arms
- Supports real-robot teleoperation with mirrored execution in simulation
- Designed for extensibility to different input devices

### 3. Data Collection Module
- Trajectory recording during teleoperation
- Supports synchronized state-action logging
- Compatible with both real-world execution and simulation

### 4. Perception Module
- Sensor data abstraction (e.g., RGB, depth, proprioception)
- Unified perception interface for downstream learning components
- Designed to decouple perception from control and data logging

---

## Added Functionality

### Real-to-Sim Teleoperation
- Teleoperation commands executed on the real robot
- Corresponding actions replayed in simulation in real time
- Enables debugging and visualization without affecting the real system

### Data Replay in Simulation
- Collected real-world trajectories can be replayed in simulation
- Useful for:
  - Verifying data quality
  - Debugging control logic
  - Offline policy evaluation

---

## Future Work

- [ ] Add teleoperated grasping support in MuJoCo
---

## Relation to Original Project

This repository is based on the following project:

- HuggingFace LeRobot: https://github.com/huggingface/lerobot

This is an independent refactoring and extension effort and is not an official
HuggingFace implementation.

---

## Notes

This repository is under active development and primarily intended
for research and experimentation.