# Unitree Aliengo Robot Model for MuJoCo

This directory contains the MuJoCo XML model of the Unitree Aliengo quadruped robot, created based on the URDF model and following the structure of the Go1 XML model.

## Model Structure

The model includes:

1. **Full Mesh Model**: `xml/aliengo.xml` - A complete model using mesh files for visual representation
2. **Approximated Model**: `xml/aliengo_approx.xml` - A simplified model using primitive shapes

## Joint Structure

The Aliengo robot has 12 actuated joints:
- 4 hip joints (FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint)
- 4 thigh joints (FR_thigh_joint, FL_thigh_joint, RR_thigh_joint, RL_thigh_joint)
- 4 calf joints (FR_calf_joint, FL_calf_joint, RR_calf_joint, RL_calf_joint)

Joint limits and actuator parameters have been set according to the original URDF specifications.

## Model Conversion Details

The model was converted from URDF to MuJoCo XML format with the following considerations:

1. **Coordinate Frame**: Maintained the same coordinate frame as the URDF model
2. **Inertial Properties**: Preserved all inertial properties from the URDF model
3. **Joint Limits**: Preserved the joint limits from the URDF model
4. **Visual Appearance**: Used the original meshes for visual representation

## Usage

This model can be used with the MuJoCo physics engine for simulation, control, and reinforcement learning tasks.

## Model Information

- **Robot Mass**: 9.041 kg (trunk) + leg components
- **Robot Height**: Approximately 0.5m (standing)
- **Degrees of Freedom**: 18 (6 DoF for floating base + 12 actuated joints)
- **Actuators**: 12 position/torque controlled motors

## Reference

- Original URDF model: `urdf/aliengo.urdf`
- Reference XML model: `Go1/xml/go1.xml` 