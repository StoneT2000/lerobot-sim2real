"""Utilities for loading and applying camera calibration data."""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import sapien
from easyhec.utils.camera_conversions import ros2opencv


def load_env_config(config_path: str) -> dict:
    """
    Load environment configuration from JSON file.

    Args:
        config_path: Path to the environment config JSON file

    Returns:
        dict: Environment configuration dictionary

    Note:
        Only the extrinsics and intrinsics file paths are used from base_camera_settings.
        Any pos/target/fov values in the config are ignored.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_camera_parameters(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera extrinsics and intrinsics from configuration.

    Args:
        config: Environment configuration dictionary

    Returns:
        tuple: (extrinsic_opencv, intrinsic) matrices
            - extrinsic_opencv: 4x4 camera extrinsic matrix in OpenCV convention
            - intrinsic: 3x3 camera intrinsic matrix
    """
    # Load base camera settings
    camera_settings = config.get("base_camera_settings", {})

    # Load extrinsics (stored in ROS convention)
    extrinsic_path = camera_settings.get("extrinsics")
    if not extrinsic_path:
        raise ValueError("No extrinsics path in config")

    extrinsic_ros = np.load(extrinsic_path)

    # Convert from ROS to OpenCV convention
    extrinsic_opencv = ros2opencv(extrinsic_ros)

    # Load intrinsics
    intrinsic_path = camera_settings.get("intrinsics")
    if not intrinsic_path:
        raise ValueError("No intrinsics path in config")

    intrinsic = np.load(intrinsic_path)

    return extrinsic_opencv, intrinsic


def apply_calibration_offset(joint_config: dict, calibration_offset: dict) -> dict:
    """
    Apply calibration offset to joint configuration.

    Args:
        joint_config: Joint configuration in radians
        calibration_offset: Calibration offsets in degrees

    Returns:
        dict: Adjusted joint configuration in radians
    """
    adjusted_config = joint_config.copy()

    for joint_name, offset_deg in calibration_offset.items():
        if joint_name in adjusted_config:
            # Convert offset from degrees to radians and apply
            offset_rad = np.deg2rad(offset_deg)
            adjusted_config[joint_name] += offset_rad

    return adjusted_config


def get_camera_pose_from_extrinsic(extrinsic_opencv: np.ndarray) -> sapien.Pose:
    """
    Convert camera extrinsic matrix to Sapien Pose.

    Args:
        extrinsic_opencv: 4x4 camera extrinsic matrix in OpenCV convention

    Returns:
        sapien.Pose: Camera pose in world coordinates

    Note:
        The extrinsic matrix represents camera-to-world transform (Tc_c2w).
        Sapien expects world-to-camera for camera setup, but we return
        camera-to-world here since the camera mount expects it.
    """
    # Extract position from the last column
    position = extrinsic_opencv[:3, 3]

    # Extract rotation matrix
    rotation_matrix = extrinsic_opencv[:3, :3]

    # Convert rotation matrix to quaternion
    # Sapien uses xyzw quaternion format
    from scipy.spatial.transform import Rotation as R

    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns [x, y, z, w]

    # Create Sapien pose
    pose = sapien.Pose(p=position, q=quaternion)

    return pose


def get_fov_from_intrinsic(intrinsic: np.ndarray, image_height: int) -> float:
    """
    Calculate vertical field of view from camera intrinsic matrix.

    Args:
        intrinsic: 3x3 camera intrinsic matrix
        image_height: Image height in pixels

    Returns:
        float: Vertical field of view in radians
    """
    # Extract focal length in y direction
    fy = intrinsic[1, 1]

    # Calculate vertical FOV
    # FOV = 2 * arctan(h / (2 * fy))
    fov_y = 2 * np.arctan(image_height / (2 * fy))

    return fov_y


def load_camera_calibration(
    env_config_path: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Dict:
    """
    Load complete camera calibration data from config file.

    Args:
        env_config_path: Path to environment config JSON
        target_width: Optional target image width for intrinsic scaling
        target_height: Optional target image height for intrinsic scaling

    Returns:
        dict: Dictionary containing:
            - 'extrinsic_opencv': 4x4 extrinsic matrix
            - 'intrinsic': 3x3 intrinsic matrix (possibly scaled)
            - 'pose': Sapien Pose object
            - 'fov': Vertical field of view in radians
            - 'calibration_offset': Joint calibration offsets (if present)
    """
    # Load config
    config = load_env_config(env_config_path)

    # Load camera parameters
    extrinsic_opencv, intrinsic = load_camera_parameters(config)

    # Scale intrinsics if target resolution provided
    if target_width is not None and target_height is not None:
        # Detect original resolution from intrinsic principal point
        # Assume principal point is at image center
        orig_width = int(intrinsic[0, 2] * 2)
        orig_height = int(intrinsic[1, 2] * 2)

        if orig_width != target_width or orig_height != target_height:
            from lerobot_sim2real.utils.camera import scale_intrinsics

            intrinsic = scale_intrinsics(
                intrinsic, orig_width, orig_height, target_width, target_height
            )

    # Get camera pose
    pose = get_camera_pose_from_extrinsic(extrinsic_opencv)

    # Calculate FOV (use target height if provided, otherwise assume from intrinsic)
    if target_height is not None:
        fov = get_fov_from_intrinsic(intrinsic, target_height)
    else:
        # Assume principal point is at center
        orig_height = int(intrinsic[1, 2] * 2)
        fov = get_fov_from_intrinsic(intrinsic, orig_height)

    result = {
        "extrinsic_opencv": extrinsic_opencv,
        "intrinsic": intrinsic,
        "pose": pose,
        "fov": fov,
    }

    # Add calibration offset if present
    if "calibration_offset" in config:
        result["calibration_offset"] = config["calibration_offset"]

    return result
