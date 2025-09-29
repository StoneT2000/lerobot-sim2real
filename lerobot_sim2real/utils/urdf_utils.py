"""URDF utilities for extracting meshes and link information.

This module provides shared functionality for loading URDF files and extracting
mesh data in different formats to support both visualization and calibration workflows.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urchin import URDF
from easyhec.utils.utils_3d import merge_meshes


def get_robot_urdf_path(uid: str) -> Path:
    """Get the URDF file path for a given robot UID.

    Args:
        uid: Robot identifier ('so100' or 'so101')

    Returns:
        Path to the URDF file

    Raises:
        ValueError: If uid is not supported
    """
    if uid == "so100":
        from easyhec import ROBOT_DEFINITIONS_DIR

        return Path(ROBOT_DEFINITIONS_DIR) / "so100" / "so100.urdf"
    elif uid == "so101":
        # Use relative path from the utils module location
        return (
            Path(__file__).parent.parent / "assets" / "robots" / "so101" / "so101.urdf"
        )
    elif uid == "so101_v2":
        return (
            Path(__file__).parent.parent
            / "assets"
            / "robots"
            / "so101"
            / "so101_v2.urdf"
        )
    else:
        raise ValueError(f"Unknown robot uid: {uid}. Supported: 'so100', 'so101'")


def load_urdf(urdf_path: Union[str, Path]) -> URDF:
    """Load a URDF file.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Loaded URDF object

    Raises:
        FileNotFoundError: If URDF file does not exist
        Exception: If URDF loading fails
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")

    return URDF.load(str(urdf_path))


def extract_individual_meshes_with_origins(
    urdf: URDF,
) -> Tuple[List, List[str], List[np.ndarray]]:
    """Extract individual meshes with their visual origins from URDF.

    This function extracts each mesh separately along with its visual origin transform.
    This is useful for detailed visualization where you need to preserve the individual
    visual elements and their poses relative to the link frame.

    Args:
        urdf: Loaded URDF object

    Returns:
        Tuple containing:
        - meshes: List of individual mesh objects
        - mesh_link_names: List of link names corresponding to each mesh
        - visual_origins: List of 4x4 visual origin transforms for each mesh
    """
    meshes = []
    mesh_link_names = []
    visual_origins = []

    for link in urdf.links:
        link_meshes = []
        link_visual_origins = []

        for visual in link.visuals:
            # Skip visuals without mesh geometry
            geom = getattr(visual, "geometry", None)
            mesh_attr = getattr(geom, "mesh", None)
            if mesh_attr is None:
                continue

            # Get visual origin (pose of visual element relative to link frame)
            visual_origin = np.eye(4)
            if hasattr(visual, "origin") and visual.origin is not None:
                visual_origin = visual.origin

            # Store each mesh with its visual origin
            for mesh in mesh_attr.meshes:
                link_meshes.append(mesh)
                link_visual_origins.append(visual_origin)

        if not link_meshes:
            continue

        # For each mesh in this link, store separately with its visual origin
        for mesh, visual_origin in zip(link_meshes, link_visual_origins):
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                meshes.append(mesh)
                mesh_link_names.append(link.name)
                visual_origins.append(visual_origin)

    return meshes, mesh_link_names, visual_origins


def extract_merged_meshes_per_link(urdf: URDF) -> Tuple[List, List[str]]:
    """Extract merged meshes per link from URDF.

    This function extracts all meshes for each link and merges them into a single
    mesh per link. Visual origins are applied to each mesh before merging to ensure
    correct positioning. This is useful for calibration and optimization where you need
    a simplified representation.

    Args:
        urdf: Loaded URDF object

    Returns:
        Tuple containing:
        - meshes: List of merged mesh objects (one per link)
        - mesh_link_names: List of link names corresponding to each merged mesh
    """
    meshes = []
    mesh_link_names = []

    for link in urdf.links:
        transformed_meshes = []

        for visual in link.visuals:
            # Skip visuals without mesh geometry
            geom = getattr(visual, "geometry", None)
            mesh_attr = getattr(geom, "mesh", None)
            if mesh_attr is None:
                continue

            # Get visual origin (pose of visual element relative to link frame)
            visual_origin = np.eye(4)
            if hasattr(visual, "origin") and visual.origin is not None:
                visual_origin = visual.origin

            # Apply visual origin transformation to each mesh before merging
            for mesh in mesh_attr.meshes:
                if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                    # Transform mesh vertices by visual origin
                    vertices = mesh.vertices
                    vertices_homo = np.hstack(
                        [vertices, np.ones((vertices.shape[0], 1))]
                    )
                    transformed_vertices = (visual_origin @ vertices_homo.T).T[:, :3]

                    # Create a new mesh with transformed vertices
                    import trimesh

                    transformed_mesh = trimesh.Trimesh(
                        vertices=transformed_vertices,
                        faces=mesh.faces,
                        process=False,  # Don't auto-process to preserve exact geometry
                    )
                    transformed_meshes.append(transformed_mesh)

        if not transformed_meshes:
            continue

        # Merge all transformed meshes for this link
        merged = merge_meshes(transformed_meshes)
        if merged is None:
            continue

        # Keep only valid triangle meshes
        if hasattr(merged, "vertices") and hasattr(merged, "faces"):
            meshes.append(merged)
            mesh_link_names.append(link.name)

    return meshes, mesh_link_names


def extract_joint_info(urdf: URDF) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """Extract joint information from URDF.

    Args:
        urdf: Loaded URDF object

    Returns:
        Tuple containing:
        - joint_limits: Dict mapping joint names to {'lower': float, 'upper': float}
        - current_config: Dict mapping joint names to initial positions (0.0)
    """
    joint_limits = {}
    current_config = {}

    for joint_name, joint in urdf.joint_map.items():
        if hasattr(joint, "limit") and joint.limit is not None:
            joint_limits[joint_name] = {
                "lower": float(joint.limit.lower)
                if joint.limit.lower is not None
                else -np.pi,
                "upper": float(joint.limit.upper)
                if joint.limit.upper is not None
                else np.pi,
            }
        else:
            # Default limits for joints without explicit limits
            joint_limits[joint_name] = {"lower": -np.pi, "upper": np.pi}

        # Initialize to zero
        current_config[joint_name] = 0.0

    return joint_limits, current_config


def get_urdf_info(urdf: URDF, num_meshes: int, mesh_link_names: List[str]) -> Dict:
    """Get summary information about the URDF.

    Args:
        urdf: Loaded URDF object
        num_meshes: Number of extracted meshes
        mesh_link_names: List of mesh link names

    Returns:
        Dictionary with URDF information
    """
    return {
        "num_links": len(urdf.links),
        "num_joints": len(urdf.joint_map),
        "num_meshes": num_meshes,
        "mesh_links": mesh_link_names,
        "joints": list(urdf.joint_map.keys()),
    }


# Convenience functions that combine loading and extraction


def load_robot_meshes_for_visualization(
    uid: str,
) -> Tuple[
    URDF, List, List[str], List[np.ndarray], Dict[str, Dict], Dict[str, float], Dict
]:
    """Load robot URDF and extract meshes for visualization.

    This is a convenience function that combines URDF loading and mesh extraction
    for visualization purposes (individual meshes with visual origins).

    Args:
        uid: Robot identifier ('so100' or 'so101')

    Returns:
        Tuple containing:
        - urdf: Loaded URDF object
        - meshes: List of individual mesh objects
        - mesh_link_names: List of link names corresponding to each mesh
        - visual_origins: List of 4x4 visual origin transforms for each mesh
        - joint_limits: Dict mapping joint names to limits
        - current_config: Dict mapping joint names to initial positions
        - info: Summary information about the URDF
    """
    urdf_path = get_robot_urdf_path(uid)
    urdf = load_urdf(urdf_path)
    meshes, mesh_link_names, visual_origins = extract_individual_meshes_with_origins(
        urdf
    )
    joint_limits, current_config = extract_joint_info(urdf)
    info = get_urdf_info(urdf, len(meshes), mesh_link_names)

    return (
        urdf,
        meshes,
        mesh_link_names,
        visual_origins,
        joint_limits,
        current_config,
        info,
    )


def load_robot_meshes_for_calibration(uid: str) -> Tuple[URDF, List, List[str]]:
    """Load robot URDF and extract merged meshes for calibration.

    This is a convenience function that combines URDF loading and mesh extraction
    for calibration purposes (merged meshes per link).

    Args:
        uid: Robot identifier ('so100' or 'so101')

    Returns:
        Tuple containing:
        - urdf: Loaded URDF object
        - meshes: List of merged mesh objects (one per link)
        - mesh_link_names: List of link names corresponding to each merged mesh
    """
    urdf_path = get_robot_urdf_path(uid)
    urdf = load_urdf(urdf_path)
    meshes, mesh_link_names = extract_merged_meshes_per_link(urdf)

    return urdf, meshes, mesh_link_names


def load_custom_urdf_for_visualization(
    urdf_path: Union[str, Path],
) -> Tuple[
    URDF, List, List[str], List[np.ndarray], Dict[str, Dict], Dict[str, float], Dict
]:
    """Load custom URDF and extract meshes for visualization.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Tuple containing:
        - urdf: Loaded URDF object
        - meshes: List of individual mesh objects
        - mesh_link_names: List of link names corresponding to each mesh
        - visual_origins: List of 4x4 visual origin transforms for each mesh
        - joint_limits: Dict mapping joint names to limits
        - current_config: Dict mapping joint names to initial positions
        - info: Summary information about the URDF
    """
    urdf = load_urdf(urdf_path)
    meshes, mesh_link_names, visual_origins = extract_individual_meshes_with_origins(
        urdf
    )
    joint_limits, current_config = extract_joint_info(urdf)
    info = get_urdf_info(urdf, len(meshes), mesh_link_names)

    return (
        urdf,
        meshes,
        mesh_link_names,
        visual_origins,
        joint_limits,
        current_config,
        info,
    )


def load_custom_urdf_for_calibration(
    urdf_path: Union[str, Path],
) -> Tuple[URDF, List, List[str]]:
    """Load custom URDF and extract merged meshes for calibration.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Tuple containing:
        - urdf: Loaded URDF object
        - meshes: List of merged mesh objects (one per link)
        - mesh_link_names: List of link names corresponding to each merged mesh
    """
    urdf = load_urdf(urdf_path)
    meshes, mesh_link_names = extract_merged_meshes_per_link(urdf)

    return urdf, meshes, mesh_link_names
