#!/usr/bin/env python3
"""
Render robot URDF from learned camera extrinsics/intrinsics (headless).

Loads camera parameters from env_config.json and renders the robot URDF
from that viewpoint, saving the output image for debugging/verification.
"""

import json
import numpy as np
import torch
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict
import tyro

# Import from existing codebase
from lerobot_sim2real.utils.urdf_utils import load_robot_meshes_for_calibration
from lerobot_sim2real.utils.camera import scale_intrinsics
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.utils.camera_conversions import ros2opencv


@dataclass
class Args:
    """Arguments for rendering robot URDF from learned camera view."""

    env_config_path: str = "env_config.json"
    """Path to environment config JSON file containing camera settings."""

    output_dir: str = "results/debug_so101_extrinsic"
    """Directory to save rendered debug image."""

    robot_urdf: str = "so101"
    """Robot type to load (so101, so100, etc.)."""

    joint_config: Optional[Dict[str, float]] = None
    """Optional joint configuration in radians. If None, uses neutral pose."""

    image_width: int = 128
    """Output image width in pixels."""

    image_height: int = 128
    """Output image height in pixels."""

    background_color: list = field(default_factory=lambda: [255, 255, 255])
    """RGB background color as [R, G, B]."""

    robot_color: list = field(default_factory=lambda: [0, 120, 255])
    """RGB robot color as [R, G, B]. Default is bright blue for good visibility."""

    output_filename: str = "rendered_view.png"
    """Output filename for rendered image."""

    apply_calibration_offset: bool = True
    """Whether to apply calibration offset from env_config.json."""


def load_env_config(config_path: str) -> dict:
    """Load environment configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_camera_parameters(config: dict) -> tuple:
    """
    Load camera extrinsics and intrinsics from paths in config.

    Returns:
        extrinsic_opencv: 4x4 camera extrinsic matrix in OpenCV convention
        intrinsic: 3x3 camera intrinsic matrix
    """
    base_camera = config["base_camera_settings"]

    # Load extrinsics (stored in ROS convention)
    extrinsic_ros = np.load(base_camera["extrinsics"])
    extrinsic_opencv = ros2opencv(extrinsic_ros)

    # Load intrinsics - try the specified path first, fallback if not found
    intrinsic_path = base_camera["intrinsics"]
    if not Path(intrinsic_path).exists():
        # Try fallback to base intrinsic file
        fallback_path = str(Path(intrinsic_path).parent / "camera_intrinsic.npy")
        if Path(fallback_path).exists():
            print(f"Warning: {intrinsic_path} not found, using {fallback_path}")
            intrinsic_path = fallback_path
        else:
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")

    intrinsic = np.load(intrinsic_path)

    return extrinsic_opencv, intrinsic


def get_neutral_joint_config(robot_urdf) -> dict:
    """Get neutral/zero joint configuration for robot."""
    cfg = {}
    for joint_name in robot_urdf.joint_map.keys():
        cfg[joint_name] = 0.0
    return cfg


def apply_calibration_offset(joint_config: dict, calibration_offset: dict) -> dict:
    """Apply calibration offsets to joint configuration."""
    adjusted_config = joint_config.copy()
    for joint_name, offset in calibration_offset.items():
        if joint_name in adjusted_config:
            # Offsets are typically in degrees, convert to radians
            adjusted_config[joint_name] += np.deg2rad(offset)
    return adjusted_config


def render_robot_from_camera(
    meshes,
    mesh_link_names,
    link_poses,
    extrinsic,
    intrinsic,
    image_width,
    image_height,
    background_color,
    robot_color,
    device,
):
    """
    Render robot meshes from camera viewpoint.

    Args:
        meshes: List of trimesh objects for robot links
        mesh_link_names: List of link names corresponding to meshes
        link_poses: Dict mapping link names to 4x4 pose matrices
        extrinsic: 4x4 camera extrinsic matrix (OpenCV convention)
        intrinsic: 3x3 camera intrinsic matrix
        image_width: Output image width
        image_height: Output image height
        background_color: RGB background color [R, G, B]
        robot_color: RGB robot color [R, G, B]
        device: torch device (cuda/cpu)

    Returns:
        rendered_image: RGB numpy array of shape (H, W, 3)
    """
    # Initialize renderer
    renderer = NVDiffrastRenderer(image_height, image_width)

    # Convert to torch tensors
    extrinsic_t = torch.from_numpy(extrinsic).float().to(device)
    intrinsic_t = torch.from_numpy(intrinsic).float().to(device)

    # IMPORTANT: The stored extrinsic is camera-to-world (Tc_c2w)
    # For rendering, we need world-to-camera (view matrix): Tw_w2c
    # So we invert it
    view_matrix = torch.linalg.inv(extrinsic_t)

    # Convert meshes to torch tensors
    link_vertices = [
        torch.from_numpy(mesh.vertices).float().to(device) for mesh in meshes
    ]
    link_faces = [torch.from_numpy(mesh.faces).int().to(device) for mesh in meshes]

    # Render each link and accumulate masks
    combined_mask = torch.zeros((image_height, image_width), device=device)

    total_pixels = 0
    for i, link_name in enumerate(mesh_link_names):
        if link_name not in link_poses:
            print(f"Warning: Link {link_name} not found in poses, skipping")
            continue

        # Get link pose (world-to-link transform)
        link_pose = torch.from_numpy(link_poses[link_name]).float().to(device)

        # Compose world-to-camera with world-to-link to get camera-to-link (view-to-model)
        # This is: T_view_model = T_world2camera @ T_world2link
        composed = view_matrix @ link_pose

        # Render mask for this link
        link_mask = renderer.render_mask(
            link_vertices[i],
            link_faces[i],
            intrinsic_t,
            composed,
        )

        # Accumulate into combined mask
        link_mask = link_mask.detach()
        link_pixels = (link_mask > 0).sum().item()
        if link_pixels > 0:
            print(f"  {link_name}: {link_pixels} pixels")
            total_pixels += link_pixels
        combined_mask[link_mask > 0] = 1

    print(f"Total pixels rendered: {total_pixels}/{image_height * image_width}")

    # Convert mask to RGB image
    mask_np = combined_mask.cpu().numpy()

    # Create RGB image with background and robot colors
    image = np.ones((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color  # Set background

    # Apply robot color where mask is True
    robot_pixels = mask_np > 0
    image[robot_pixels] = robot_color

    return image


def main(args: Args):
    """Main function to render robot from learned camera view."""

    print(f"Loading environment config from {args.env_config_path}")
    config = load_env_config(args.env_config_path)

    print("Loading camera parameters...")
    extrinsic, intrinsic = load_camera_parameters(config)

    print(f"Camera extrinsic (OpenCV):\n{extrinsic}")
    print(f"Camera intrinsic (original):\n{intrinsic}")

    # Detect original resolution from intrinsic principal point
    # Assume principal point is at image center for resolution estimation
    orig_width = int(intrinsic[0, 2] * 2)
    orig_height = int(intrinsic[1, 2] * 2)

    # Scale intrinsics if output resolution differs from intrinsic resolution
    if orig_width != args.image_width or orig_height != args.image_height:
        print(
            f"Scaling intrinsics from {orig_width}x{orig_height} to {args.image_width}x{args.image_height}"
        )
        intrinsic = scale_intrinsics(
            intrinsic, orig_width, orig_height, args.image_width, args.image_height
        )
        print(f"Camera intrinsic (scaled):\n{intrinsic}")

    # Load robot URDF and meshes
    print(f"Loading robot URDF: {args.robot_urdf}")
    robot_urdf, meshes, mesh_link_names = load_robot_meshes_for_calibration(
        args.robot_urdf
    )
    print(f"Loaded {len(meshes)} links: {mesh_link_names}")

    # Setup joint configuration
    if args.joint_config is not None:
        joint_config = args.joint_config
        print(f"Using provided joint config: {joint_config}")
    else:
        joint_config = get_neutral_joint_config(robot_urdf)
        print("Using neutral joint configuration")

    # Apply calibration offset if specified in config and flag is enabled
    if args.apply_calibration_offset and "calibration_offset" in config:
        print("Applying calibration offsets...")
        print(f"  Offsets: {config['calibration_offset']}")
        joint_config = apply_calibration_offset(
            joint_config, config["calibration_offset"]
        )
    elif "calibration_offset" in config:
        print("Skipping calibration offsets (--apply-calibration-offset=False)")

    # Compute forward kinematics to get link poses
    print("Computing forward kinematics...")
    # Ensure all joints in URDF have values (default to 0 if not specified)
    cfg = {}
    for joint_name in robot_urdf.joint_map.keys():
        cfg[joint_name] = joint_config.get(joint_name, 0.0)

    link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
    print(f"Computed poses for {len(link_poses)} links")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Render robot from camera view
    print("Rendering robot from camera view...")
    rendered_image = render_robot_from_camera(
        meshes=meshes,
        mesh_link_names=mesh_link_names,
        link_poses=link_poses,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_width=args.image_width,
        image_height=args.image_height,
        background_color=args.background_color,
        robot_color=args.robot_color,
        device=device,
    )

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename

    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)

    print(f"✓ Rendered image saved to: {output_path}")
    print(f"  Image size: {args.image_width}x{args.image_height}")
    print(f"  Background: RGB{tuple(args.background_color)}")
    print(f"  Robot: RGB{tuple(args.robot_color)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
