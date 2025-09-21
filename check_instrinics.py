import numpy as np

K = np.load("results/so101/so101_follower/base_camera/camera_intrinsic.npy")
print("K=\n", K, "shape=", K.shape, "dtype=", K.dtype)

# Use your masks/images to get H, W
masks = np.load("results/so101/so101_follower/base_camera/mask.npy")
H, W = masks.shape[1], masks.shape[2]
cx, cy = float(K[0, 2]), float(K[1, 2])
fx, fy = float(K[0, 0]), float(K[1, 1])
print(f"Resolution: {W}x{H}")
print(f"Principal point offset: dx={cx - W / 2:.2f}, dy={cy - H / 2:.2f}")

# FOV from K (in degrees)
import math

hfov = 2 * math.degrees(math.atan(0.5 * W / fx))
vfov = 2 * math.degrees(math.atan(0.5 * H / fy))
print(f"HFOV≈{hfov:.2f}°, VFOV≈{vfov:.2f}°")
