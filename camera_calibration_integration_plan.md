# Camera Calibration Integration Plan

## Overview

This plan outlines the integration of learned camera extrinsics and intrinsics from the calibration process into the SO101 grasp cube environment.

## Current State Analysis

### render_urdf_from_learned_camera.py

- Loads camera parameters from `env_config.json`
- Contains utilities for:
  - Loading environment config
  - Loading camera extrinsics (ROS convention) and intrinsics from .npy files
  - Converting ROS to OpenCV convention
  - Scaling intrinsics for different resolutions
  - Applying calibration offsets to joint configurations

### grasp_cube.py

- Currently uses hardcoded camera settings in `base_camera_settings`
- Supports domain randomization for camera poses
- Camera configuration happens in:
  - `_default_sensor_configs` property
  - `sample_camera_poses` method
  - Camera mount initialization in `_load_scene`

## Proposed Changes

### 1. Extract Shared Utilities to utils/

Create new utility modules:

#### `lerobot_sim2real/utils/calibration.py`

```python
"""Utilities for loading and applying camera calibration data."""

- load_env_config(config_path: str) -> dict
- load_camera_parameters(config: dict) -> tuple[np.ndarray, np.ndarray]
- apply_calibration_offset(joint_config: dict, calibration_offset: dict) -> dict
- get_camera_pose_from_extrinsic(extrinsic_opencv: np.ndarray) -> sapien.Pose
```

#### Update `lerobot_sim2real/utils/camera.py`

Add to existing file:

```python
- scale_intrinsics(intrinsic, orig_width, orig_height, new_width, new_height)
  (move from render script if not already there)
```

### 2. Update SO101GraspCubeEnv

#### Constructor Updates

Add new parameters:

```python
def __init__(
    self,
    *args,
    env_config_path: Optional[str] = None,
    use_learned_camera: bool = False,
    override_camera_settings: Optional[dict] = None,
    **kwargs
):
```

#### New Methods

```python
def _load_camera_from_config(self, config_path: str) -> dict:
    """Load camera settings from calibration config."""

def _setup_camera_from_calibration(self) -> tuple[sapien.Pose, float, np.ndarray]:
    """Setup camera pose and intrinsics from calibration data."""
```

#### Update Existing Methods

**`_default_sensor_configs`**

- Check if `use_learned_camera` is True
- If yes, load camera parameters from config
- Set camera pose from extrinsics
- Set FOV from intrinsics
- Handle resolution scaling if needed

**`_load_scene`**

- Apply calibration offsets to robot if available in config
- Set camera mount to calibrated pose if not using domain randomization

**`sample_camera_poses`**

- If `use_learned_camera` and not domain randomizing, return fixed calibrated pose
- If domain randomizing, use calibrated pose as center for randomization

### 3. Domain Randomization Configuration Updates

Update `SO101GraspCubeDomainRandomizationConfig`:

```python
@dataclass
class SO101GraspCubeDomainRandomizationConfig:
    # ... existing fields ...

    ### Calibration-related randomization ###
    randomize_around_calibrated_camera: bool = False
    """If True, randomize camera around calibrated pose instead of base_camera_settings"""

    apply_calibration_offset_noise: bool = False
    """If True, add noise to calibration offsets"""

    calibration_offset_noise_scale: float = 0.01
    """Scale of noise added to calibration offsets (in radians)"""
```

### 4. Integration Workflow

1. **Calibration Phase**:

   - Run camera calibration to generate `env_config.json`
   - Save camera extrinsics and intrinsics as .npy files

2. **Training Phase**:

   ```python
   env = SO101GraspCubeEnv(
       env_config_path="env_config.json",
       use_learned_camera=True,
       domain_randomization_config={
           "randomize_around_calibrated_camera": True,
           # other DR settings
       }
   )
   ```

3. **Evaluation Phase**:
   ```python
   env = SO101GraspCubeEnv(
       env_config_path="env_config.json",
       use_learned_camera=True,
       domain_randomization=False  # No randomization for eval
   )
   ```

### 5. Backward Compatibility

- Default behavior remains unchanged (use `base_camera_settings`)
- Only use calibration if `env_config_path` is provided and `use_learned_camera=True`
- Allow `override_camera_settings` to manually specify camera params

### 6. Testing Strategy

1. **Unit Tests**:

   - Test calibration loading utilities
   - Test camera pose conversion from extrinsics
   - Test intrinsic scaling

2. **Integration Tests**:

   - Test environment with and without calibration
   - Test domain randomization around calibrated pose
   - Test calibration offset application

3. **Visual Verification**:
   - Render scenes with calibrated camera
   - Compare with `render_urdf_from_learned_camera.py` output
   - Verify robot appears in expected location

### 7. Implementation Steps

1. **Phase 1: Extract Utilities** (Priority: High)

   - Create `calibration.py` utility module
   - Move shared functions from render script
   - Add unit tests

2. **Phase 2: Basic Integration** (Priority: High)

   - Add calibration loading to environment
   - Update camera configuration
   - Test with fixed calibrated camera

3. **Phase 3: Domain Randomization** (Priority: Medium)

   - Add randomization around calibrated pose
   - Add calibration offset noise
   - Test training stability

4. **Phase 4: Documentation** (Priority: Medium)
   - Update environment documentation
   - Add calibration workflow guide
   - Create example scripts

### 8. Configuration File Format

Expected `env_config.json` structure:

```json
{
  "base_camera_settings": {
    "extrinsics": "path/to/camera_extrinsic.npy",
    "intrinsics": "path/to/camera_intrinsic.npy"
  },
  "calibration_offset": {
    "joint1": 0.1,
    "joint2": -0.05
    // ... other joint offsets in degrees
  }
}
```

### 9. Benefits

1. **Accurate Sim2Real Transfer**: Use exact camera parameters from real setup
2. **Flexible Training**: Can randomize around calibrated pose for robustness
3. **Easy Evaluation**: Direct comparison with real robot setup
4. **Modular Design**: Utilities can be reused for other environments

### 10. Potential Challenges

1. **Coordinate System Conversions**: Ensure correct ROS to OpenCV conversions
2. **Resolution Handling**: Properly scale intrinsics for different training resolutions
3. **Backward Compatibility**: Maintain existing functionality
4. **Performance**: Minimize overhead of loading calibration data

## 11. Testing with SO101 Calibration Test Script

### Updates to `so101_sim2real_calibration_test.py`

The existing calibration test script can be enhanced to verify the learned camera parameters:

#### New Features to Add

1. **Camera View Testing Mode**

   ```python
   class SO101CalibrationTester:
       def __init__(self, test_camera_view=False, env_config_path=None):
           # ... existing code ...
           self.test_camera_view = test_camera_view
           self.env_config_path = env_config_path or "env_config.json"
   ```

2. **Setup Simulation with Learned Camera**

   ```python
   def setup_simulation(self):
       """Setup ManiSkill simulation environment"""
       # Note: render_mode controls the visualization backend:
       # - "human": Opens interactive viewer window
       # - "rgb_array": Returns RGB frames (for recording)
       # - "cameras": Returns all camera views
       #
       # For testing camera calibration, we want BOTH:
       # 1. Interactive viewer to see robot movement (render_mode="human")
       # 2. Camera observations to verify calibration (obs_mode="rgb+segmentation")

       env_kwargs = dict(
           obs_mode="rgb+segmentation" if self.test_camera_view else "none",
           control_mode="pd_joint_pos",
           sim_backend="physx_cpu",
           render_mode="human",  # Keep interactive viewer for joint testing
       )

       # Add camera calibration parameters if testing camera view
       if self.test_camera_view and Path(self.env_config_path).exists():
           env_kwargs.update(
               env_config_path=self.env_config_path,
               use_learned_camera=True,
               domain_randomization=False,  # No DR for testing
           )
   ```

   **Design Note**: The original suggestion to use `render_mode="cameras"` was incorrect.
   The `render_mode` parameter in ManiSkill controls the rendering backend:

   - `"human"`: Interactive viewer window
   - `"rgb_array"`: Offscreen rendering for recording
   - `"cameras"`: Not a standard mode

   Instead, camera observations should be controlled via:

   - `obs_mode`: Determines what observations are returned ("rgb", "rgbd", "rgb+segmentation", etc.)
   - Environment-specific camera configs in `_default_sensor_configs`

3. **Camera View Comparison**

   ```python
   def setup_camera_view_comparison(self):
       """Setup side-by-side view: learned camera vs default camera"""
       if not self.test_camera_view:
           return

       # Create figure for visualization
       self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
       self.ax1.set_title("Learned Camera View")
       self.ax2.set_title("Default Camera View")
       plt.ion()
       plt.show()

   def update_camera_views(self):
       """Update camera view comparison display"""
       if not self.test_camera_view:
           return

       # Get observation from environment
       obs = self.sim_env.get_obs()
       if "rgb" in obs and "base_camera" in obs["rgb"]:
           learned_view = obs["rgb"]["base_camera"]
           self.ax1.clear()
           self.ax1.imshow(learned_view)
           self.ax1.set_title("Learned Camera View")
           self.ax1.axis('off')

       # Optionally render with default camera for comparison
       # This would require temporarily switching camera settings

       plt.draw()
       plt.pause(0.001)
   ```

4. **Render Comparison with Reference Image**

   ```python
   def load_reference_render(self):
       """Load reference render from render_urdf_from_learned_camera.py output"""
       ref_path = Path(self.env_config_path).parent / "rendered_view.png"
       if ref_path.exists():
           self.reference_image = plt.imread(str(ref_path))
           return True
       return False

   def compare_with_reference(self):
       """Show reference render alongside live view"""
       if hasattr(self, 'reference_image'):
           self.ax2.clear()
           self.ax2.imshow(self.reference_image)
           self.ax2.set_title("Reference Render")
           self.ax2.axis('off')
   ```

5. **Extended Calibration Loop**

   ```python
   def run_calibration_loop(self):
       """Main calibration loop with camera testing"""
       # ... existing setup ...

       if self.test_camera_view:
           self.setup_camera_view_comparison()
           self.load_reference_render()

       while self.running:
           # ... existing joint reading/setting ...

           # Update camera views if in camera test mode
           if self.test_camera_view:
               self.update_camera_views()
               self.compare_with_reference()

           # ... rest of loop ...
   ```

### Usage Examples

1. **Basic Joint Calibration Test** (existing functionality):

   ```bash
   python so101_sim2real_calibration_test.py
   ```

2. **Camera View Test**:

   ```bash
   python so101_sim2real_calibration_test.py --test-camera-view --env-config-path env_config.json
   ```

3. **Full Integration Test**:

   ```bash
   # First render reference image
   python render_urdf_from_learned_camera.py --env-config-path env_config.json

   # Then test with live robot
   python so101_sim2real_calibration_test.py --test-camera-view --env-config-path env_config.json
   ```

### Verification Checklist

When running camera view tests, verify:

1. **Camera Position**: Robot appears in same location as reference render
2. **Camera Angle**: View angle matches reference
3. **Joint Offsets**: Robot pose matches when at rest position
4. **Field of View**: Image boundaries show same scene extent
5. **Image Quality**: Resolution and aspect ratio are correct

### Benefits of Integration

1. **Real-time Verification**: See how calibration affects robot appearance
2. **Visual Debugging**: Quickly spot misalignments or errors
3. **Training Preview**: Verify what the policy will see during training
4. **Documentation**: Generate comparison images for reports

## Next Steps

1. Review and approve plan
2. Create utility modules
3. Implement basic integration
4. Test with existing calibration data using updated calibration test script
5. Extend with domain randomization features
