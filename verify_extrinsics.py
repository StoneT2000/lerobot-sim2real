#!/usr/bin/env python3
"""Verify the extrinsic matrix relationship"""

import numpy as np

# Our calibrated extrinsic (camera-to-world)
calibrated_extrinsic = np.array([
    [-0.96609509, -0.00464125, -0.2581445, 0.44641343],
    [0.25356442, 0.17127877, -0.95203394, 0.65203542],
    [0.04863329, -0.98521167, -0.16429472, 0.26752639],
    [0., 0., 0., 1.]
])

# What ManiSkill shows as extrinsic_cv
maniskill_extrinsic_cv = np.array([
    [0.9520, 0.1713, -0.2536, -0.4688],
    [0.2581, -0.0046, 0.9661, -0.3707],
    [0.1643, -0.9852, -0.0486, 0.5821]
])

# What ManiSkill shows as cam2world_gl
maniskill_cam2world_gl = np.array([
    [0.9520, -0.2581, -0.1643, 0.4464],
    [0.1713, 0.0046, 0.9852, 0.6520],
    [-0.2536, -0.9661, 0.0486, 0.2675],
    [0.0000, 0.0000, 0.0000, 1.0000]
])

print("Calibrated extrinsic (camera-to-world):")
print(calibrated_extrinsic)
print()

print("ManiSkill extrinsic_cv (3x4):")
print(maniskill_extrinsic_cv)
print()

print("ManiSkill cam2world_gl (4x4):")
print(maniskill_cam2world_gl)
print()

# Check if extrinsic_cv is the inverse
calibrated_inv = np.linalg.inv(calibrated_extrinsic)
print("Inverse of calibrated extrinsic (world-to-camera):")
print(calibrated_inv[:3, :])
print()

# Compare
print("Difference between maniskill_extrinsic_cv and inverse[:3,:]:")
diff = maniskill_extrinsic_cv - calibrated_inv[:3, :]
print(np.max(np.abs(diff)))
print()

# Check the relationship between cam2world_gl and calibrated_extrinsic
print("Comparing cam2world_gl with calibrated_extrinsic:")
print("Position match:", np.allclose(maniskill_cam2world_gl[:3, 3], calibrated_extrinsic[:3, 3]))
print("Rotation similarity:")
print("  calibrated R:")
print(calibrated_extrinsic[:3, :3])
print("  cam2world R:")
print(maniskill_cam2world_gl[:3, :3])

# The issue might be that cam2world_gl uses a different convention (OpenGL vs OpenCV)
# OpenGL: Y-up, -Z forward
# OpenCV: Z-forward, Y-down
# Let's check if there's a coordinate system transformation

# Try transposing the rotation part
print("\nTransposed cam2world rotation:")
print(maniskill_cam2world_gl[:3, :3].T)
print("\nDoes it match calibrated rotation?", np.allclose(maniskill_cam2world_gl[:3, :3].T, calibrated_extrinsic[:3, :3], atol=0.01))
