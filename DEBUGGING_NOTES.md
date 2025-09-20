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
