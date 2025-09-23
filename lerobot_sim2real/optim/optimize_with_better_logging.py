from typing import List, Optional

import torch
from tqdm import tqdm

from easyhec.optim.rb_solver import RBSolver, RBSolverConfig


def optimize(
    initial_extrinsic_guess: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    masks: torch.Tensor,
    link_poses_dataset: torch.Tensor,
    meshes: List[str],
    camera_width: int,
    camera_height: int,
    camera_mount_poses: Optional[torch.Tensor] = None,
    iterations: int = 5000,
    learning_rate: float = 3e-3,
    gt_camera_pose: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    early_stopping_steps: int = 200,
    verbose: bool = True,
    return_history: bool = False,
):
    """
    Optimizes an initial guess of a camera extrinsic using the camera intrinsic matrix, a dataset of robot masks, link poses relative to the robot base frame, and paths to the mesh files of each of the link poses.

    Inputs are expected to be torch tensors on the same device. If they are on the GPU, all optimization will be done on the GPU. Poses are expected to follow the opencv2 transformation conventions.


    Parameters:

        initial_extrinsic_guess (torch.Tensor, shape (4, 4)): Initial guess of the camera extrinsic
        camera_intrinsic (torch.Tensor, shape (3, 3)): Camera intrinsic matrix
        masks (torch.Tensor, shape (N, H, W)): Robot segmentation masks
        link_poses_dataset (torch.Tensor, shape (N, L, 4, 4)): Link poses relative to any frame (e.g. the robot base frame), where N is the number of samples, L is the number of links
        meshes (List[str | trimesh.Trimesh]): List of paths to the mesh files of each of the L links. Can also be a list of trimesh.Trimesh objects.
        camera_width (int): Camera width
        camera_height (int): Camera height
        camera_mount_poses (torch.Tensor, shape (N, 4, 4)): Used for cameras that are fixed relative to some mount that may be moving. If None, then the camera is assumed to be fixed. If provided the initial extrinsic guess should be relative to the mount frame.
        iterations (int): Number of optimization iterations
        learning_rate (float): Learning rate for the Adam optimizer
        batch_size (int): Default is None meaning whole batch optimization. Otherwise this specifies the number of samples to process in each batch.
        gt_camera_pose (torch.Tensor, shape (4, 4)): Default is None. If a ground truth camera pose is provided the optimization function will compute error metrics relative to the ground truth camera pose.
        early_stopping_steps (int): Default is 200. If the loss has not improved after this many steps the optimization will stop.
        verbose (bool): Default is True. If True, will print the loss value and a progress bar.
        return_history (bool): Default is False. If True, will return a list of all the current and previous best predicted extrinsics.
    """
    device = initial_extrinsic_guess.device
    cfg = RBSolverConfig(
        camera_width=camera_width,
        camera_height=camera_height,
        robot_masks=masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        initial_extrinsic_guess=initial_extrinsic_guess,
    )
    solver = RBSolver(cfg)
    solver = solver.to(device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=learning_rate)
    best_predicted_extrinsic = initial_extrinsic_guess.clone()
    best_loss = float("inf")
    last_loss_improvement_step = 0
    pbar = tqdm(range(iterations)) if verbose else range(iterations)
    dataset = dict(
        intrinsic=camera_intrinsic,
        link_poses=link_poses_dataset,
        mask=masks,
        mount_poses=camera_mount_poses,
    )
    if gt_camera_pose is not None:
        dataset["gt_camera_pose"] = gt_camera_pose

    if return_history:
        best_extrinsics = []
        best_extrinsics_steps = []
        best_extrinsics_losses = []
    for i in pbar:
        if batch_size is None:
            batch = dataset
        else:
            bid = torch.randperm(len(dataset["mask"]))[:batch_size]
            batch = {k: v[bid] for k, v in dataset.items()}
        output = solver(batch)
        optimizer.zero_grad()
        output["mask_loss"].backward()
        optimizer.step()
        loss_value = output["mask_loss"].item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_predicted_extrinsic = solver.get_predicted_extrinsic()
            last_loss_improvement_step = i
            if return_history:
                best_extrinsics.append(best_predicted_extrinsic)
                best_extrinsics_steps.append(i)
                best_extrinsics_losses.append(loss_value)
        if i - last_loss_improvement_step >= early_stopping_steps:
            break
        if verbose:
            pbar.set_description(f"Loss: {loss_value:.2f}, Best Loss: {best_loss:.2f}")
        if "metrics" in output:
            pbar.set_postfix(output["metrics"])
    if return_history:
        # return torch.stack(extrinsics)

        return {
            "best_extrinsics": torch.stack(best_extrinsics),
            "best_extrinsics_step": best_extrinsics_steps,
            "best_extrinsics_losses": best_extrinsics_losses,
        }
    else:
        return best_predicted_extrinsic
