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
    mask_color: tuple = (255, 0, 0),  # Red color in RGB
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
        mask = torch.zeros((camera_height, camera_width), device=device)
        for j, link_pose in enumerate(link_poses_dataset[i]):
            link_mask = renderer.render_mask(
                link_vertices[j],
                link_faces[j],
                intrinsic,
                camera_pose @ link_pose,
            )
            link_mask = link_mask.detach()
            mask[link_mask > 0] = 1
        return mask

    def apply_red_mask_overlay(image, mask, alpha=0.6):
        """Apply red mask overlay to image"""
        result = image.copy()
        mask_3d = np.stack([mask, mask, mask], axis=2)
        red_overlay = np.full_like(image, mask_color)
        result = (1 - alpha) * result + alpha * red_overlay
        result = result * mask_3d + image * (1 - mask_3d)
        return result.astype(np.uint8)

    for i in tqdm(range(len(images))):
        overlaid_images = []
        for j in range(len(extrinsics)):
            if camera_mount_poses is not None:
                mask = get_mask_from_camera_pose(extrinsics[j] @ camera_mount_poses[i])
            else:
                mask = get_mask_from_camera_pose(extrinsics[j])
            mask = mask.cpu().numpy()
            overlaid_images.append(apply_red_mask_overlay(images[i], mask))

        num_subplots = len(extrinsics) + 1 if masks is not None else len(extrinsics)

        plt.rcParams.update(
            {"font.size": 16}
        )  # Increase font size for all text elements
        fig = plt.figure(figsize=(7 * num_subplots, 8))
        for j in range(len(extrinsics)):
            ax = fig.add_subplot(1, num_subplots, j + 1)
            ax.imshow(overlaid_images[j])
            ax.axis("off")
            ax.set_title(labels[j])

        if masks is not None:
            ax = fig.add_subplot(1, num_subplots, num_subplots)
            reference_mask = apply_red_mask_overlay(images[i], masks[i])
            ax.imshow(reference_mask)
            ax.axis("off")
            ax.set_title("Masks")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{i}.png")
        plt.close()
        if return_rgb:
            return cv2.cvtColor(cv2.imread(f"{output_dir}/{i}.png"), cv2.COLOR_BGR2RGB)
