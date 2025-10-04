#!/usr/bin/env python3
"""Test different coordinate system interpretations"""

import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R
from easyhec.utils.camera_conversions import ros2opencv

# Load the calibration data
extrinsic_ros_path = "results/so101/so101_follower/base_camera/camera_extrinsic_ros.npy"
extrinsic_ros = np.load(extrinsic_ros_path)

print("ROS Extrinsic matrix:")
print(extrinsic_ros)
print()

# Convert to OpenCV
extrinsic_opencv = ros2opencv(extrinsic_ros)
print("OpenCV Extrinsic matrix (via ros2opencv):")
print(extrinsic_opencv)
print()

# Try using ROS directly (without conversion)
print("Testing different interpretations:")
print()

# Method 1: Use OpenCV as-is
pos1 = extrinsic_opencv[:3, 3]
rot1 = R.from_matrix(extrinsic_opencv[:3, :3])
quat1 = rot1.as_quat()
print(f"Method 1 (OpenCV as-is):")
print(f"  Position: {pos1}")
print(f"  Quaternion: {quat1}")
print()

# Method 2: Use ROS as-is
pos2 = extrinsic_ros[:3, 3]
rot2 = R.from_matrix(extrinsic_ros[:3, :3])
quat2 = rot2.as_quat()
print(f"Method 2 (ROS as-is):")
print(f"  Position: {pos2}")
print(f"  Quaternion: {quat2}")
print()

# Method 3: Sapien uses a different convention - try inverting
extrinsic_inv = np.linalg.inv(extrinsic_opencv)
pos3 = extrinsic_inv[:3, 3]
rot3 = R.from_matrix(extrinsic_inv[:3, :3])
quat3 = rot3.as_quat()
print(f"Method 3 (Inverted OpenCV):")
print(f"  Position: {pos3}")
print(f"  Quaternion: {quat3}")
print()

# Method 4: Try a different axis conversion (Sapien might use Y-up)
# Convert from Z-forward to Y-up by rotating -90 degrees around X
rot_x_neg90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
extrinsic_yup = extrinsic_opencv.copy()
extrinsic_yup[:3, :3] = extrinsic_opencv[:3, :3] @ rot_x_neg90.T
pos4 = extrinsic_yup[:3, 3]
rot4 = R.from_matrix(extrinsic_yup[:3, :3])
quat4 = rot4.as_quat()
print(f"Method 4 (Y-up conversion):")
print(f"  Position: {pos4}")
print(f"  Quaternion: {quat4}")
print()

# Check what the environment is using
print("Current environment values:")
print(f"  Position: [0.44641342759132385, 0.6520354151725769, 0.26752638816833496]")
print(f"  Quaternion: [0.08203779, 0.75856256, -0.6384594, -0.10110497]")

# Compare with the methods
print("\nClosest match:")
current_pos = np.array([0.44641342759132385, 0.6520354151725769, 0.26752638816833496])
current_quat = np.array([0.08203779, 0.75856256, -0.6384594, -0.10110497])

for i, (pos, quat) in enumerate([(pos1, quat1), (pos2, quat2), (pos3, quat3), (pos4, quat4)]):
    pos_diff = np.linalg.norm(pos - current_pos)
    quat_diff = min(np.linalg.norm(quat - current_quat), np.linalg.norm(quat + current_quat))  # Handle quaternion sign ambiguity
    print(f"Method {i+1}: pos_diff={pos_diff:.6f}, quat_diff={quat_diff:.6f}")
