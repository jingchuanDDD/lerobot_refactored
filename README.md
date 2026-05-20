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

In addition, new features are introduced to support:
- Real-robot teleoperation with synchronized simulation replay
- Data replay in simulation for policy debugging and evaluation

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
- [ ] Extend teleoperation to more complex manipulation tasks
- [ ] Integrate learning-based policies for closed-loop evaluation

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