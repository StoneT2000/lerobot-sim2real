# DEBUGGING NOTES

Yes: IoU is 0.0000 for every step. That means the rendered mask never overlaps your GT mask at all; the optimizer is likely minimizing area by moving the robot out of frame (loss drops a bit, then plateaus at a constant = GT mask area).
Your Tw_w2c starts exactly at [-0.4000, 0.1000, 0.5000] and only drifts slightly while loss flattens. That tiny drift with zero IoU suggests a pose-convention or composition mismatch rather than just learning-rate issues.
Two concrete issues to fix/check:
Preview overlay isn’t applying the new invert/swap flags: \_render_mask_from_extrinsic in web_easyhec_camera_calibration.py still does extrinsic_t @ link_poses_t[j] unconditionally. So the preview visualization can disagree with the solver/visualizer. Apply the invert/swap flags there.
Validate that the renderer comp expects camera-to-link. In your original solver (from easyhec) it passed Tc_c2l = Tc_c2b @ ... @ link_pose. In our override we render with Tw_w2c @ Tw2l which is mathematically the same camera-to-link transform, but if the engine expects the other convention, you’ll get disjoint masks. Quick test: try removing the inverse and use Tc_c2b @ Tw2l for one run; if IoU > 0 immediately, the convention was flipped.
Fast sanity checks:
Print masks_ref[0].sum() and rendered_masks[0].sum() once; if the latter is ~0, you’re off-screen.
Try preview with: invert OFF, swap ON. If IoU jumps above 0, you’ve confirmed the order/convention mismatch.
Recheck intrinsics (fx, fy, cx, cy) printed values and ensure NVDiffrastRenderer(H, W) matches (H, W) ordering expected by your renderer.

### SO101-specific checks that matter

- **Joint zero alignment**:

  - Tune `CALIBRATION_OFFSET` for SO101. Even if limits are similar, SO101’s actual zero vs URDF zero can differ, especially at elbow/shoulder due to different joint frames.
  - If a joint consistently looks mirrored in overlays, introduce a sign map (e.g., `JOINT_SIGN["shoulder_pan"] = -1`) before building `cfg`.

- **Joint axes and frames**:

  - SO101 uses mostly `axis="0 0 1"` with non-trivial joint `origin rpy`. If your bus angles were implicitly matched to SO100’s axes, you may need a sign flip on one or two joints for SO101.

- **qpos samples within limits**:

  - Current samples are conservative, but verify against SO101 limits (e.g., `wrist_roll` ≈ ±2.93 rad). If a sample violates a limit in practice, use slightly smaller magnitudes to avoid edge cases.

- **Initial extrinsic guess**:

  - Because the SO101 base frame differs from SO100, adjust the initial guess if overlays look globally shifted/tilted. The current guess is reasonable; try nudging yaw/height if optimization stalls.

- **Mesh silhouette differences**:

  - The gripper geometry is different. Re-generate masks for SO101; don’t reuse SO100 masks. A mismatch will force the optimizer to “explain” geometry with a wrong camera pose.

- **FK configuration build**:
  - The `cfg` uses `urdf.joint_map` and defaults others to 0, which is correct. Ensure the six servo joints from the bus match those joint names (they do for SO101).
