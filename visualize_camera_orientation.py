#!/usr/bin/env python3
"""Visualize camera orientation to debug the issue"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Load calibration data
extrinsic_path = "results/so101/so101_follower/base_camera/camera_extrinsic_opencv.npy"
extrinsic_opencv = np.load(extrinsic_path)

print("Calibrated extrinsic matrix:")
print(extrinsic_opencv)

# Extract camera position and orientation
cam_pos = extrinsic_opencv[:3, 3]
cam_rot = extrinsic_opencv[:3, :3]

# Camera coordinate axes in world frame
cam_x = cam_rot[:, 0]  # Right
cam_y = cam_rot[:, 1]  # Down
cam_z = cam_rot[:, 2]  # Forward (viewing direction)

print(f"\nCamera position: {cam_pos}")
print(f"Camera X axis (right): {cam_x}")
print(f"Camera Y axis (down): {cam_y}")
print(f"Camera Z axis (forward): {cam_z}")

# Default camera from env_config.json
default_pos = np.array([0.69, 0.37, 0.28])
default_target = np.array([0.185, -0.15, 0.0])

# Compute default camera orientation (look_at)
default_forward = default_target - default_pos
default_forward = default_forward / np.linalg.norm(default_forward)
# Assume Y is up in world frame
world_up = np.array([0, 0, 1])
default_right = np.cross(default_forward, world_up)
default_right = default_right / np.linalg.norm(default_right)
default_up = np.cross(default_right, default_forward)

print(f"\nDefault camera position: {default_pos}")
print(f"Default camera forward: {default_forward}")
print(f"Default camera right: {default_right}")
print(f"Default camera up: {default_up}")

# Create 3D plot
fig = plt.figure(figsize=(12, 6))

# Plot 1: Calibrated camera
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Calibrated Camera")

# Draw camera axes
scale = 0.1
ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_x[0], cam_x[1], cam_x[2], 
           length=scale, color='red', label='X (right)')
ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_y[0], cam_y[1], cam_y[2], 
           length=scale, color='green', label='Y (down)')
ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2], cam_z[0], cam_z[1], cam_z[2], 
           length=scale*2, color='blue', label='Z (forward)')

# Draw line to approximate target
approx_target = cam_pos + cam_z * 0.5
ax1.plot([cam_pos[0], approx_target[0]], 
         [cam_pos[1], approx_target[1]], 
         [cam_pos[2], approx_target[2]], 'b--', alpha=0.5)

# Draw robot approximate position (origin)
ax1.scatter([0], [0], [0], color='black', s=100, label='Robot base')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax1.set_xlim([-0.5, 1])
ax1.set_ylim([-0.5, 1])
ax1.set_zlim([0, 1])

# Plot 2: Default camera
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Default Camera")

# Draw camera axes
ax2.quiver(default_pos[0], default_pos[1], default_pos[2], 
           default_right[0], default_right[1], default_right[2], 
           length=scale, color='red', label='X (right)')
ax2.quiver(default_pos[0], default_pos[1], default_pos[2], 
           -default_up[0], -default_up[1], -default_up[2],  # Y down in camera frame
           length=scale, color='green', label='Y (down)')
ax2.quiver(default_pos[0], default_pos[1], default_pos[2], 
           default_forward[0], default_forward[1], default_forward[2], 
           length=scale*2, color='blue', label='Z (forward)')

# Draw line to target
ax2.plot([default_pos[0], default_target[0]], 
         [default_pos[1], default_target[1]], 
         [default_pos[2], default_target[2]], 'b--', alpha=0.5)

# Draw robot approximate position (origin)
ax2.scatter([0], [0], [0], color='black', s=100, label='Robot base')
ax2.scatter(default_target[0], default_target[1], default_target[2], 
            color='orange', s=50, label='Target')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()
ax2.set_xlim([-0.5, 1])
ax2.set_ylim([-0.5, 1])
ax2.set_zlim([0, 1])

plt.tight_layout()
plt.savefig("camera_orientations.png", dpi=150)
print("\nSaved visualization to camera_orientations.png")

# Check if cameras are looking in similar directions
print(f"\nCalibrated camera forward direction: {cam_z}")
print(f"Default camera forward direction: {default_forward}")
print(f"Dot product (similarity): {np.dot(cam_z, default_forward):.3f}")
print("(1.0 = same direction, 0.0 = perpendicular, -1.0 = opposite)")

# Compute where the calibrated camera is looking
calibrated_target = cam_pos + cam_z * 0.5
print(f"\nCalibrated camera is looking toward: {calibrated_target}")
print(f"Default camera is looking toward: {default_target}")
