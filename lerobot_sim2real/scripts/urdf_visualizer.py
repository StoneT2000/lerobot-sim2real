#!/usr/bin/env python3
"""
URDF Visualizer using Gradio
Visualizes robot URDF files in the browser with interactive joint controls.
"""

import numpy as np
import gradio as gr
from typing import Dict, List, Optional, Tuple
from urchin import URDF

# Import shared URDF utilities
from lerobot_sim2real.utils.urdf_utils import (
    load_custom_urdf_for_visualization,
    load_custom_urdf_for_calibration,
    get_robot_urdf_path,
)

try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")


class URDFVisualizer:
    def __init__(self):
        self.robot_urdf: Optional[URDF] = None
        # Individual meshes with visual origins (for detailed visualization)
        self.meshes: List = []
        self.mesh_link_names: List[str] = []
        self.visual_origins: List = []  # Store visual origin transforms
        # Merged meshes per link (for calibration)
        self.merged_meshes: List = []
        self.merged_mesh_link_names: List[str] = []
        self.joint_limits: Dict = {}
        self.current_config: Dict = {}

    def load_urdf(self, urdf_path: str) -> Tuple[str, Dict]:
        """Load URDF file and extract meshes."""
        try:
            # Use shared utility function to load URDF and extract meshes for visualization
            (
                self.robot_urdf,
                self.meshes,
                self.mesh_link_names,
                self.visual_origins,
                self.joint_limits,
                self.current_config,
                info,
            ) = load_custom_urdf_for_visualization(urdf_path)

            # Also load merged meshes for calibration comparison
            (
                _,  # We already have the URDF
                self.merged_meshes,
                self.merged_mesh_link_names,
            ) = load_custom_urdf_for_calibration(urdf_path)

            # Update info to include merged mesh count
            info["num_merged_meshes"] = len(self.merged_meshes)
            info["merged_mesh_links"] = self.merged_mesh_link_names

            return (
                f"Successfully loaded URDF: {len(self.meshes)} individual meshes, {len(self.merged_meshes)} merged meshes from {len(self.robot_urdf.links)} links",
                info,
            )

        except Exception as e:
            return f"Error loading URDF: {str(e)}", {}

    def update_joint_config(self, **joint_values) -> Tuple[go.Figure, go.Figure]:
        """Update joint configuration and return updated 3D plots."""
        if self.robot_urdf is None:
            return self.create_empty_plot(), self.create_empty_plot()

        # Update configuration with provided joint values
        for joint_name, value in joint_values.items():
            if joint_name in self.current_config:
                self.current_config[joint_name] = float(value)

        return self.create_3d_plot(), self.create_merged_mesh_plot()

    def create_3d_plot(self):
        """Create 3D plotly figure of the robot."""
        if not PLOTLY_AVAILABLE:
            return self.create_empty_plot()
        if self.robot_urdf is None or not self.meshes:
            return self.create_empty_plot()

        fig = go.Figure()

        try:
            # Compute forward kinematics
            link_poses = self.robot_urdf.link_fk(
                cfg=self.current_config, use_names=True
            )

            # Color palette for different links
            colors = px.colors.qualitative.Set3

            for i, (mesh, link_name, visual_origin) in enumerate(
                zip(self.meshes, self.mesh_link_names, self.visual_origins)
            ):
                if link_name not in link_poses:
                    continue

                # Get link pose and combine with visual origin
                link_pose = link_poses[link_name]
                # Apply both link pose and visual origin: link_pose * visual_origin
                final_pose = link_pose @ visual_origin

                # Transform mesh vertices
                if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                    vertices = mesh.vertices
                    faces = mesh.faces

                    # Apply combined transformation (link pose + visual origin)
                    vertices_homo = np.hstack(
                        [vertices, np.ones((vertices.shape[0], 1))]
                    )
                    transformed_vertices = (final_pose @ vertices_homo.T).T[:, :3]

                    # Create mesh trace
                    color = colors[i % len(colors)]
                    fig.add_trace(
                        go.Mesh3d(
                            x=transformed_vertices[:, 0],
                            y=transformed_vertices[:, 1],
                            z=transformed_vertices[:, 2],
                            i=faces[:, 0],
                            j=faces[:, 1],
                            k=faces[:, 2],
                            color=color,
                            opacity=0.8,
                            name=f"{link_name}_{i}",  # Make names unique per mesh
                            showlegend=True,
                        )
                    )

            # Add coordinate frame at origin
            self._add_coordinate_frame(fig, np.eye(4), size=0.1)

        except Exception as e:
            print(f"Error creating 3D plot: {e}")
            return self.create_empty_plot()

        # Update layout with proper aspect ratio
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",  # Use actual data proportions
                aspectratio=dict(x=1, y=1, z=1),  # Force equal aspect ratios
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2),
                    center=dict(x=0, y=0, z=0.3),  # Look at center of robot workspace
                    up=dict(x=0, y=0, z=1),  # Z-up coordinate system
                ),
            ),
            title="URDF Robot Visualization",
            showlegend=True,
            width=800,
            height=800,  # Make it square to avoid stretching
        )

        return fig

    def create_merged_mesh_plot(self):
        """Create 3D plotly figure of the robot using merged meshes (calibration view)."""
        if not PLOTLY_AVAILABLE:
            return self.create_empty_plot()
        if self.robot_urdf is None or not self.merged_meshes:
            return self.create_empty_plot()

        fig = go.Figure()

        try:
            # Compute forward kinematics
            link_poses = self.robot_urdf.link_fk(
                cfg=self.current_config, use_names=True
            )

            # Color palette for different links (darker colors for merged view)
            colors = px.colors.qualitative.Dark2

            for i, (mesh, link_name) in enumerate(
                zip(self.merged_meshes, self.merged_mesh_link_names)
            ):
                if link_name not in link_poses:
                    continue

                # Get link pose (visual origins already applied during mesh merging)
                link_pose = link_poses[link_name]

                # Transform mesh vertices
                if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                    vertices = mesh.vertices
                    faces = mesh.faces

                    # Apply link transformation
                    vertices_homo = np.hstack(
                        [vertices, np.ones((vertices.shape[0], 1))]
                    )
                    transformed_vertices = (link_pose @ vertices_homo.T).T[:, :3]

                    # Create mesh trace with darker color for merged view
                    color = colors[i % len(colors)]
                    fig.add_trace(
                        go.Mesh3d(
                            x=transformed_vertices[:, 0],
                            y=transformed_vertices[:, 1],
                            z=transformed_vertices[:, 2],
                            i=faces[:, 0],
                            j=faces[:, 1],
                            k=faces[:, 2],
                            color=color,
                            opacity=0.9,  # Slightly more opaque for merged view
                            name=f"{link_name}_merged",
                            showlegend=True,
                        )
                    )

            # Add coordinate frame at origin
            self._add_coordinate_frame(fig, np.eye(4), size=0.1)

        except Exception as e:
            print(f"Error creating merged mesh plot: {e}")
            return self.create_empty_plot()

        # Update layout with proper aspect ratio
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",  # Use actual data proportions
                aspectratio=dict(x=1, y=1, z=1),  # Force equal aspect ratios
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2),
                    center=dict(x=0, y=0, z=0.3),  # Look at center of robot workspace
                    up=dict(x=0, y=0, z=1),  # Z-up coordinate system
                ),
            ),
            title="Merged Meshes (Calibration View)",
            showlegend=True,
            width=800,
            height=800,  # Make it square to avoid stretching
        )

        return fig

    def _add_coordinate_frame(self, fig, pose: np.ndarray, size: float = 0.1):
        """Add coordinate frame arrows to the plot."""
        origin = pose[:3, 3]
        x_axis = pose[:3, 0] * size + origin
        y_axis = pose[:3, 1] * size + origin
        z_axis = pose[:3, 2] * size + origin

        # X axis (red)
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], x_axis[0]],
                y=[origin[1], x_axis[1]],
                z=[origin[2], x_axis[2]],
                mode="lines",
                line=dict(color="red", width=5),
                name="X-axis",
                showlegend=False,
            )
        )

        # Y axis (green)
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], y_axis[0]],
                y=[origin[1], y_axis[1]],
                z=[origin[2], y_axis[2]],
                mode="lines",
                line=dict(color="green", width=5),
                name="Y-axis",
                showlegend=False,
            )
        )

        # Z axis (blue)
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_axis[0]],
                y=[origin[1], z_axis[1]],
                z=[origin[2], z_axis[2]],
                mode="lines",
                line=dict(color="blue", width=5),
                name="Z-axis",
                showlegend=False,
            )
        )

    def create_empty_plot(self):
        """Create empty 3D plot."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            title="Load a URDF file to visualize",
            width=800,
            height=800,  # Square aspect ratio
        )
        return fig


def create_gradio_interface():
    """Create the Gradio interface."""
    visualizer = URDFVisualizer()

    # Predefined URDF paths for convenience
    default_paths = {
        "SO101 Robot": str(get_robot_urdf_path("so101")),
        "SO100 Robot": str(get_robot_urdf_path("so100")),
        "Custom Path": "",
    }

    with gr.Blocks(title="URDF Visualizer") as demo:
        gr.Markdown("# 🤖 URDF Robot Visualizer")
        gr.Markdown("Load and visualize URDF robot files with basic joint controls.")

        with gr.Row():
            with gr.Column(scale=1):
                # URDF Loading Section
                gr.Markdown("## 📁 Load URDF")
                path_dropdown = gr.Dropdown(
                    choices=list(default_paths.keys()),
                    value="SO101 Robot",
                    label="Predefined Robots",
                )
                custom_path = gr.Textbox(
                    label="Custom URDF Path",
                    placeholder="Enter path to URDF file...",
                    visible=False,
                )
                load_btn = gr.Button("Load URDF", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)

                # Robot Info Section
                gr.Markdown("## ℹ️ Robot Info")
                info_json = gr.JSON(label="Robot Information", visible=False)

                # Simple Joint Controls (fixed sliders for common robot joints)
                gr.Markdown("## 🎮 Joint Controls")
                gr.Markdown("*Adjust common robot joints (if they exist in the URDF):*")

                joint1_slider = gr.Slider(
                    -3.14, 3.14, 0, step=0.01, label="Joint 1 (shoulder_pan)"
                )
                joint2_slider = gr.Slider(
                    -3.14, 3.14, 0, step=0.01, label="Joint 2 (shoulder_lift)"
                )
                joint3_slider = gr.Slider(
                    -3.14, 3.14, 0, step=0.01, label="Joint 3 (elbow_flex)"
                )
                joint4_slider = gr.Slider(
                    -3.14, 3.14, 0, step=0.01, label="Joint 4 (wrist_flex)"
                )
                joint5_slider = gr.Slider(
                    -3.14, 3.14, 0, step=0.01, label="Joint 5 (wrist_roll)"
                )
                joint6_slider = gr.Slider(0, 1, 0.2, step=0.01, label="Gripper")

                update_pose_btn = gr.Button("Update Robot Pose", variant="secondary")
                reset_view_btn = gr.Button("Reset View", variant="secondary")

            with gr.Column(scale=3):
                # 3D Visualizations
                gr.Markdown("## 🎯 Robot Visualization")
                with gr.Row():
                    with gr.Column():
                        plot_3d_detailed = gr.Plot(
                            label="Detailed View (Individual Meshes)",
                            value=visualizer.create_empty_plot(),
                        )
                    with gr.Column():
                        plot_3d_merged = gr.Plot(
                            label="Calibration View (Merged Meshes)",
                            value=visualizer.create_empty_plot(),
                        )

        def update_path_visibility(selected):
            if selected == "Custom Path":
                return gr.update(visible=True), ""
            else:
                return gr.update(visible=False), default_paths.get(selected, "")

        def load_urdf_handler(selected_robot, custom_path_val):
            if selected_robot == "Custom Path":
                urdf_path = custom_path_val
            else:
                urdf_path = default_paths.get(selected_robot, "")

            if not urdf_path:
                return (
                    "Please provide a URDF path",
                    {},
                    visualizer.create_empty_plot(),
                    visualizer.create_empty_plot(),
                )

            status, info = visualizer.load_urdf(urdf_path)

            # Create initial plots
            detailed_plot = visualizer.create_3d_plot()
            merged_plot = visualizer.create_merged_mesh_plot()

            return status, info, detailed_plot, merged_plot

        def update_robot_pose(j1, j2, j3, j4, j5, gripper):
            """Update robot pose based on joint slider values."""
            if visualizer.robot_urdf is None:
                return visualizer.create_empty_plot(), visualizer.create_empty_plot()

            # Map common joint names to slider values
            joint_mapping = {
                "shoulder_pan": j1,
                "shoulder_lift": j2,
                "elbow_flex": j3,
                "wrist_flex": j4,
                "wrist_roll": j5,
                "gripper": gripper,
            }

            # Only update joints that exist in the URDF
            joint_updates = {}
            for joint_name, value in joint_mapping.items():
                if joint_name in visualizer.current_config:
                    joint_updates[joint_name] = value

            return visualizer.update_joint_config(**joint_updates)

        def reset_view_handler():
            """Reset the 3D view to default camera position."""
            return visualizer.create_3d_plot(), visualizer.create_merged_mesh_plot()

        # Event handlers
        path_dropdown.change(
            update_path_visibility,
            inputs=[path_dropdown],
            outputs=[custom_path, custom_path],
        )

        load_btn.click(
            load_urdf_handler,
            inputs=[path_dropdown, custom_path],
            outputs=[status_text, info_json, plot_3d_detailed, plot_3d_merged],
        )

        update_pose_btn.click(
            update_robot_pose,
            inputs=[
                joint1_slider,
                joint2_slider,
                joint3_slider,
                joint4_slider,
                joint5_slider,
                joint6_slider,
            ],
            outputs=[plot_3d_detailed, plot_3d_merged],
        )

        reset_view_btn.click(
            reset_view_handler,
            outputs=[plot_3d_detailed, plot_3d_merged],
        )

        # Auto-update on slider change (optional - can be slow)
        for slider in [
            joint1_slider,
            joint2_slider,
            joint3_slider,
            joint4_slider,
            joint5_slider,
            joint6_slider,
        ]:
            slider.change(
                update_robot_pose,
                inputs=[
                    joint1_slider,
                    joint2_slider,
                    joint3_slider,
                    joint4_slider,
                    joint5_slider,
                    joint6_slider,
                ],
                outputs=[plot_3d_detailed, plot_3d_merged],
            )

    return demo


if __name__ == "__main__":
    # Check dependencies
    if not PLOTLY_AVAILABLE:
        print("❌ Missing required dependency: plotly")
        print("Please install with: pip install plotly")
        exit(1)

    try:
        from urchin import URDF
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install urchin trimesh")
        exit(1)

    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=1337, share=False)
