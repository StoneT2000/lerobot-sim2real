# URDF Rendering from Learned Camera Extrinsics

## Overview

This document describes the `render_urdf_from_learned_camera.py` script, which renders a robot URDF from a learned camera viewpoint for debugging and verification purposes. It also documents findings related to jaw/gripper misalignment due to calibration offsets.

## Script Purpose

The script loads camera extrinsics and intrinsics learned during camera calibration (stored in `env_config.json`) and renders the robot URDF from that camera's viewpoint. This is useful for:

- **Verifying camera calibration results** - Check if learned extrinsics produce reasonable robot projections
- **Debugging sim2real alignment** - Visualize how the robot appears from the calibrated camera view
- **Understanding calibration offsets** - See the effect of joint calibration offsets on robot appearance

## Installation & Requirements

The script requires the following dependencies (already in the project):

- `numpy`
- `torch`
- `opencv-python` (cv2)
- `tyro`
- `trimesh`
- `urchin`
- `easyhec` (for NVDiffrastRenderer)

Ensure your virtual environment is activated:

```bash
source .venv/bin/activate
```

## Basic Usage

### Default Rendering

Renders the robot in neutral pose with default colors:

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py
```

**Output:** `results/debug_so101_extrinsic/rendered_view.png`

### High Contrast Rendering

For better visibility, use bright colors on dark background:

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --background-color 0 0 0 \
  --robot-color 0 255 0 \
  --output-filename robot_green.png
```

### Custom Joint Configuration

Render with specific joint angles (in radians):

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --joint-config shoulder_pan 0.0 shoulder_lift -0.5 elbow_flex 1.0 \
  --background-color 0 0 0 \
  --robot-color 255 0 0 \
  --output-filename custom_pose.png
```

### Different Image Resolution

Change output resolution (also scales intrinsics automatically):

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --image-width 256 \
  --image-height 256 \
  --output-filename high_res.png
```

## Calibration Offset Control

### What are Calibration Offsets?

Calibration offsets compensate for differences between:

- Where the robot's motors define "zero position"
- Where the URDF defines "zero position"

These offsets are stored in `env_config.json` under `calibration_offset` and are **critical for sim2real alignment**.

### Rendering WITH Calibration Offsets (Default)

Shows how the real robot actually appears:

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --background-color 0 0 0 \
  --robot-color 0 255 0 \
  --output-filename with_offset.png
```

### Rendering WITHOUT Calibration Offsets

Shows the ideal URDF pose (useful for debugging):

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --no-apply-calibration-offset \
  --background-color 0 0 0 \
  --robot-color 0 255 0 \
  --output-filename no_offset.png
```

## Jaw/Gripper Misalignment Findings

### Issue Description

When rendering the SO101 robot with learned camera extrinsics, the gripper jaw may appear to have an "overbit" or misalignment. This is **expected behavior** when calibration offsets are applied.

### Root Cause

The gripper calibration offset in `env_config.json`:

```json
"calibration_offset": {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": 0,
    "gripper": -55
}
```

The **-55 degree offset** on the gripper joint causes the jaw to appear rotated/misaligned when viewed in the rendered output.

### Why This Offset Exists

1. **Motor Calibration Mismatch** - The physical robot's motor "zero" position doesn't match the URDF's defined zero position
2. **Sim2Real Alignment** - This offset is necessary to make the simulated robot match the real robot's pose
3. **Intentional Correction** - The offset was likely tuned during calibration to achieve proper sim2real correspondence

### Diagnostic Process

1. **Render with offset** (default):

   ```bash
   python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
     --background-color 0 0 0 --robot-color 0 255 0 \
     --output-filename with_calibration.png
   ```

2. **Render without offset**:

   ```bash
   python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
     --no-apply-calibration-offset \
     --background-color 0 0 0 --robot-color 0 255 0 \
     --output-filename no_calibration.png
   ```

3. **Compare the images** to see the effect of the -55° gripper offset

### Interpretation

- **With offset (default)** → Shows how the **real robot** appears from camera view
- **Without offset** → Shows the **ideal URDF** pose (may not match reality)

The jaw misalignment is **correct** if it matches what you see in the real camera feed. If the real robot also shows this jaw position, the calibration is working as intended.

### Solutions/Workarounds

#### Option 1: Visual Compensation (for debugging only)

Manually counter the offset when rendering for visualization:

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --joint-config gripper 0.96 \
  --background-color 0 0 0 --robot-color 0 255 0 \
  --output-filename compensated.png
```

Note: 0.96 radians ≈ 55 degrees (counters the -55° offset)

#### Option 2: Recalibrate Gripper Offset

If the jaw looks wrong in **both** simulation and reality, recalibrate:

1. Run the calibration script again
2. Adjust the gripper offset in `env_config.json`
3. Re-verify with this rendering script

#### Option 3: Disable Offset for Pure Visualization

For clean visualizations without real-world corrections:

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --no-apply-calibration-offset
```

## Configuration File

The script reads from `env_config.json` (or specify with `--env-config-path`):

```json
{
    "base_camera_settings": {
        "extrinsics": "results/so101/so101_follower/base_camera/camera_extrinsic_ros.npy",
        "intrinsics": "results/so101/so101_follower/base_camera/camera_intrinsic_128x128.npy",
        ...
    },
    "calibration_offset": {
        "shoulder_pan": 0,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": 0,
        "gripper": -55
    }
}
```

### Automatic Fallbacks

- If `camera_intrinsic_128x128.npy` doesn't exist, falls back to `camera_intrinsic.npy`
- Intrinsics are automatically scaled to match output resolution
- Extrinsics are automatically converted from ROS to OpenCV coordinate system

## Command-Line Arguments

| Argument                     | Type | Default                         | Description                                        |
| ---------------------------- | ---- | ------------------------------- | -------------------------------------------------- |
| `--env-config-path`          | str  | `env_config.json`               | Path to environment config JSON                    |
| `--output-dir`               | str  | `results/debug_so101_extrinsic` | Output directory for images                        |
| `--robot-urdf`               | str  | `so101`                         | Robot type (so101, so100, etc.)                    |
| `--joint-config`             | dict | None                            | Joint angles in radians (e.g., `shoulder_pan 0.5`) |
| `--image-width`              | int  | 128                             | Output image width                                 |
| `--image-height`             | int  | 128                             | Output image height                                |
| `--background-color`         | list | `[255, 255, 255]`               | RGB background color                               |
| `--robot-color`              | list | `[0, 120, 255]`                 | RGB robot color                                    |
| `--output-filename`          | str  | `rendered_view.png`             | Output filename                                    |
| `--apply-calibration-offset` | bool | True                            | Apply calibration offsets from config              |

## Output Interpretation

### Debug Information

The script prints useful diagnostic info:

```
Loading camera parameters...
Camera extrinsic (OpenCV):
[[-0.966 -0.005 -0.258  0.446]
 [ 0.254  0.171 -0.952  0.652]
 [ 0.049 -0.985 -0.164  0.268]
 [ 0.000  0.000  0.000  1.000]]

Scaling intrinsics from 640x491 to 128x128
Applying calibration offsets...
  Offsets: {'gripper': -55, ...}

Rendering robot from camera view...
  base_link: 367 pixels
  shoulder_link: 229 pixels
  ...
Total pixels rendered: 1568/16384
```

### Pixel Count Analysis

- **0 pixels** → Robot is completely out of view (check extrinsics)
- **Few pixels (< 500)** → Robot is barely visible (might be far away or at edge)
- **Many pixels (> 1000)** → Good visibility from camera view ✓

## Troubleshooting

### Blank White Image

**Problem:** Robot not rendering, only white background
**Causes:**

1. Extrinsic matrix needs inversion (fixed in current version)
2. Robot is completely out of camera view
3. Intrinsics are incorrect

**Solutions:**

- Check pixel count in output (should be > 0)
- Try higher contrast colors: `--background-color 0 0 0 --robot-color 255 0 0`
- Verify extrinsics file exists and is valid

### Robot Appears Wrong

**Problem:** Robot pose doesn't match expectations
**Causes:**

1. Calibration offsets changing joint positions
2. Wrong joint configuration
3. Coordinate system mismatch

**Solutions:**

- Compare with/without calibration: `--no-apply-calibration-offset`
- Check joint config: provide explicit `--joint-config`
- Verify extrinsics are in correct coordinate system (ROS → OpenCV conversion)

### File Not Found

**Problem:** `camera_intrinsic_128x128.npy` not found
**Solution:** Script automatically falls back to `camera_intrinsic.npy` and scales it

## Examples

### Example 1: Quick Verification

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --background-color 0 0 0 \
  --robot-color 0 255 0
```

### Example 2: Compare Calibration Effect

```bash
# With calibration
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --background-color 0 0 0 --robot-color 0 255 0 \
  --output-filename with_offset.png

# Without calibration
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --no-apply-calibration-offset \
  --background-color 0 0 0 --robot-color 0 255 0 \
  --output-filename no_offset.png
```

### Example 3: Specific Pose

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --joint-config shoulder_lift -0.785 elbow_flex 1.57 wrist_flex 1.57 \
  --background-color 0 0 0 --robot-color 255 128 0 \
  --output-filename reaching_pose.png
```

### Example 4: High Resolution

```bash
python lerobot_sim2real/scripts/render_urdf_from_learned_camera.py \
  --image-width 512 --image-height 512 \
  --background-color 255 255 255 --robot-color 50 50 200 \
  --output-filename high_res_render.png
```

## Technical Details

### Coordinate Systems

- **Input Extrinsics:** ROS convention (Z-forward, X-right, Y-down)
- **Rendering:** OpenCV convention (Z-forward, X-right, Y-down)
- **Conversion:** Automatic via `ros2opencv()` utility

### Rendering Pipeline

1. Load camera extrinsics (camera-to-world) and intrinsics
2. Invert extrinsics to get view matrix (world-to-camera)
3. Compute forward kinematics for robot joints
4. For each link: compose view matrix with link pose
5. Render mask using NVDiffrastRenderer
6. Accumulate all link masks
7. Convert to RGB image with specified colors

### Frame Transforms

```
T_view_model = T_world2camera @ T_world2link
where:
  T_world2camera = inv(T_camera2world)  # Loaded extrinsic inverted
  T_world2link = robot_urdf.link_fk()    # From forward kinematics
```

## Related Scripts

- **`easyhec_camera_calibration.py`** - Learns the camera extrinsics
- **`web_easyhec_camera_calibration.py`** - Web UI for calibration
- **`urdf_visualizer.py`** - Interactive 3D URDF viewer (Gradio)
- **`tune_calibration_offset.py`** - Adjust calibration offsets

## Summary

The `render_urdf_from_learned_camera.py` script is a valuable debugging tool for verifying camera calibration and understanding sim2real alignment. The jaw misalignment observed is an expected consequence of the -55° gripper calibration offset, which is necessary for matching the simulated robot to the real robot's physical configuration.

**Key Takeaway:** If the rendered robot (with offsets) matches what you see in the real camera, your calibration is working correctly!
