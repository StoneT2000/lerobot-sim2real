# 🤖 URDF Visualizer

A web-based URDF robot visualizer built with Gradio and Plotly. Visualize robot URDF files in your browser with interactive 3D rendering and joint controls.

## Features

- 🌐 **Web-based**: Runs in your browser, works over SSH/remote connections
- 🎮 **Interactive**: Real-time joint control with sliders
- 📊 **3D Visualization**: High-quality 3D mesh rendering with Plotly
- 🔧 **Easy Setup**: Simple installation and launch scripts
- 🤖 **Robot-Ready**: Pre-configured for SO101/SO100 robots

## Quick Start

### 1. Install Dependencies (if needed)

```bash
pip3 install gradio plotly numpy trimesh urchin
```

### 2. Launch the Visualizer

```bash
# Simple launch (auto-installs dependencies)
./run_urdf_visualizer.sh

# Or run directly
python3 urdf_visualizer.py
```

### 3. Open in Browser

Navigate to: `http://localhost:1337`

## Usage

### Loading a URDF

1. **Predefined Robots**: Select "SO101 Robot" from the dropdown for the built-in robot
2. **Custom URDF**: Select "Custom Path" and enter the path to your URDF file
3. Click **"Load URDF"** to visualize

### Joint Controls

- Use the 6 joint sliders to control common robot joints:
  - Joint 1: `shoulder_pan`
  - Joint 2: `shoulder_lift`
  - Joint 3: `elbow_flex`
  - Joint 4: `wrist_flex`
  - Joint 5: `wrist_roll`
  - Gripper: `gripper`
- Sliders auto-update the robot pose in real-time
- Only joints that exist in your URDF will be affected

### 3D Visualization

- **Mouse Controls**:
  - Rotate: Click and drag
  - Zoom: Scroll wheel
  - Pan: Right-click and drag
- **Coordinate Frame**: Red (X), Green (Y), Blue (Z) axes shown at origin
- **Mesh Colors**: Each robot link gets a different color for easy identification

## Supported Formats

- **URDF Files**: Standard ROS URDF format
- **Mesh Files**: STL, OBJ, PLY, and other formats supported by trimesh
- **Coordinate Systems**: ROS coordinate conventions

## Configuration

### Default Paths

Edit the `default_paths` dictionary in `urdf_visualizer.py` to add your own robot presets:

```python
default_paths = {
    "SO101 Robot": "/path/to/so101.urdf",
    "Your Robot": "/path/to/your_robot.urdf",
    "Custom Path": ""
}
```

### Joint Limits

The visualizer automatically reads joint limits from the URDF file. If no limits are specified, it defaults to ±π radians.

## Troubleshooting

### Common Issues

1. **"URDF file not found"**

   - Check the file path is correct and accessible
   - Ensure the URDF file exists and is readable

2. **"No meshes found"**

   - Verify your URDF references valid mesh files
   - Check that mesh file paths in the URDF are correct

3. **"Import errors"**

   - Install missing dependencies: `pip3 install gradio plotly numpy trimesh urchin`

4. **Slow visualization**
   - Large/complex meshes can slow down rendering
   - Consider simplifying meshes or reducing detail

### Remote Access

To access over SSH/remote connection:

1. Launch with: `python3 urdf_visualizer.py`
2. Access via: `http://your-server-ip:1337`
3. Or use SSH port forwarding: `ssh -L 1337:localhost:1337 user@server`

## Integration

This visualizer can be easily integrated into other projects:

```python
from urdf_visualizer import URDFVisualizer

# Create visualizer
viz = URDFVisualizer()

# Load URDF
status, info = viz.load_urdf("/path/to/robot.urdf")

# Create 3D plot
fig = viz.create_3d_plot()

# Update joint configuration
fig = viz.update_joint_config(shoulder_pan=0.5, elbow_flex=1.0)
```

## Requirements

- Python 3.7+
- gradio
- plotly
- numpy
- trimesh
- urchin (for URDF loading)
- easyhec (for mesh utilities)

## License

This tool is part of the lerobot-sim2real project.
