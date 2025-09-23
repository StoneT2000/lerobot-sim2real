import os.path as osp
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer


def visualize_extrinsic_results_red_mask(
    images,
    link_poses_dataset,
    meshes,
    intrinsic: np.ndarray,
    extrinsics: np.ndarray,
    camera_mount_poses: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    labels: List[str] = [],
    output_dir="results/",
    return_rgb: bool = False,
    mask_color: tuple = (255, 0, 0),  # Red color in RGB (fallback)
    mask_colors: Optional[List[tuple]] = None,  # Per-extrinsic colors (RGB)
    fill_alpha: float = 0.5,
    overlay_edges: bool = True,
    edge_thickness: int = 2,
    invert_extrinsic: bool = False,
    swap_multiplication_order: bool = False,
):
    """
    Visualizes a given list of extrinsic matrices and draws the mask cameras at those extrinsics would project on the original RGB images.
    Modified version that uses red color for mask overlay instead of darkening.

    Args:
        images (np.ndarray, shape (N, H, W, 3)): List of RGB images to visualize.
        link_poses_dataset (np.ndarray, shape (N, L, 4, 4)): Link poses relative to any frame (e.g. the robot base frame), where N is the number of samples, L is the number of links
        meshes (List[str | trimesh.Trimesh]): List of mesh paths or trimesh.Trimesh objects for each of the links
        intrinsic (np.ndarray, shape (3, 3)): Camera intrinsic matrix
        extrinsics (np.ndarray, shape (M, 4, 4)): Extrinsic matrices to visualize
        camera_mount_poses (np.ndarray, shape (N, 4, 4)): Camera mount poses relative to the robot base frame, where N is the number of samples. If none then camera is assumed to be fixed.
        masks (np.ndarray, shape (N, H, W)): If given, will also display an image showing the masks used for optimization on top of the original images.
        labels (List[str]): List of labels for each of the extrinsics
        output_dir (str): Directory to save the visualizations
        mask_color (tuple): RGB color for mask overlay (default: red)
    """
    ### visualization code for the predicted extrinsic ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_height, camera_width = images[0].shape[:2]
    renderer = NVDiffrastRenderer(camera_height, camera_width)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    intrinsic_t = torch.from_numpy(intrinsic).float().to(device)
    if camera_mount_poses is not None:
        camera_mount_poses = torch.from_numpy(camera_mount_poses).float().to(device)
    link_poses_dataset = torch.from_numpy(link_poses_dataset).float().to(device)

    for i in range(len(meshes)):
        if isinstance(meshes[i], str):
            meshes[i] = trimesh.load(osp.expanduser(meshes[i]), force="mesh")
    link_vertices = [mesh.vertices.copy() for mesh in meshes]
    link_faces = [mesh.faces.copy() for mesh in meshes]
    link_vertices = [
        torch.from_numpy(mesh.vertices).float().to(device) for mesh in meshes
    ]
    link_faces = [torch.from_numpy(mesh.faces).int().to(device) for mesh in meshes]

    def get_mask_from_camera_pose(camera_pose):
        cam_pose = camera_pose
        if invert_extrinsic:
            cam_pose = torch.linalg.inv(cam_pose)
        mask = torch.zeros((camera_height, camera_width), device=device)
        for j, link_pose in enumerate(link_poses_dataset[i]):
            composed = (
                cam_pose @ link_pose
                if not swap_multiplication_order
                else link_pose @ cam_pose
            )
            link_mask = renderer.render_mask(
                link_vertices[j],
                link_faces[j],
                intrinsic_t,
                composed,
            )
            link_mask = link_mask.detach()
            mask[link_mask > 0] = 1
        return mask

    def apply_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple,
        alpha: float,
        draw_edges: bool,
        edge_px: int,
    ) -> np.ndarray:
        """Apply a colored fill + optional edge outline overlay to the input image where mask==1."""
        mask_bool = mask.astype(bool)
        result = image.copy()
        if mask_bool.any():
            # Fill overlay
            if alpha > 0:
                overlay = np.full_like(result, color)
                result[mask_bool] = (
                    (1 - alpha) * result[mask_bool] + alpha * overlay[mask_bool]
                ).astype(np.uint8)
            # Edge outline for visibility
            if draw_edges:
                edges = cv2.Canny((mask_bool.astype(np.uint8) * 255), 50, 150)
                if edge_px > 1:
                    k = max(1, edge_px // 2)
                    kernel = np.ones((k, k), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                result[edges > 0] = color
        return result

    for i in tqdm(range(len(images))):
        overlaid_images = []
        # Choose per-extrinsic colors: default to blue for first, green for second, else cycle distinct hues
        default_colors = [(30, 144, 255), (0, 255, 0), (255, 215, 0), (255, 0, 0)]
        color_list = (
            mask_colors
            if mask_colors is not None and len(mask_colors) >= len(extrinsics)
            else [
                default_colors[j % len(default_colors)] for j in range(len(extrinsics))
            ]
        )

        for j in range(len(extrinsics)):
            if camera_mount_poses is not None:
                composed_extr = extrinsics[j] @ camera_mount_poses[i]
                mask = get_mask_from_camera_pose(composed_extr)
            else:
                mask = get_mask_from_camera_pose(extrinsics[j])
            mask = mask.cpu().numpy()
            overlaid_images.append(
                apply_mask_overlay(
                    images[i],
                    mask,
                    color=color_list[j],
                    alpha=fill_alpha,
                    draw_edges=overlay_edges,
                    edge_px=edge_thickness,
                )
            )

        # Arrange subplots in a grid layout with 10 items per row
        num_tiles = len(extrinsics) + (1 if masks is not None else 0)
        ncols = 10 if num_tiles >= 10 else num_tiles
        nrows = int(np.ceil(num_tiles / max(1, ncols)))

        plt.rcParams.update({"font.size": 16})
        fig = plt.figure(figsize=(6.5 * ncols, 6.5 * nrows))
        for j in range(len(extrinsics)):
            ax = fig.add_subplot(nrows, ncols, j + 1)
            ax.imshow(overlaid_images[j])
            ax.axis("off")
            title = labels[j] if j < len(labels) else f"Extrinsic {j}"
            ax.set_title(title)
            # Color the subplot border to match the overlay color for clarity
            for spine in ax.spines.values():
                spine.set_edgecolor(np.array(color_list[j]) / 255.0)
                spine.set_linewidth(3)

        if masks is not None:
            ax = fig.add_subplot(nrows, ncols, len(extrinsics) + 1)
            # Show input masks (if provided) using requested mask_color
            reference_mask = apply_mask_overlay(
                images[i],
                masks[i],
                color=mask_color,
                alpha=0.4,
                draw_edges=True,
                edge_px=2,
            )
            ax.imshow(reference_mask)
            ax.axis("off")
            ax.set_title("Masks")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{i}.png")
        plt.close()
        if return_rgb:
            return cv2.cvtColor(cv2.imread(f"{output_dir}/{i}.png"), cv2.COLOR_BGR2RGB)
