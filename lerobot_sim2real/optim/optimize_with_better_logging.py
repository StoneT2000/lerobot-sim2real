from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from lerobot_sim2real.optim.rb_solver_override import RBSolver, RBSolverConfig


def optimize(
    initial_extrinsic_guess: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    masks: torch.Tensor,
    link_poses_dataset: torch.Tensor,
    meshes: List[str],
    camera_width: int,
    camera_height: int,
    camera_mount_poses: Optional[torch.Tensor] = None,
    iterations: int = 10000,
    learning_rate: float = 3e-4,
    gt_camera_pose: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    early_stopping_steps: int = 10000,
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
    worst_loss = float("-inf")
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

    # Always initialize all_losses for logging purposes
    all_losses = []
    prev_rendered_sum = None

    if return_history:
        best_extrinsics = []
        best_extrinsics_steps = []
        best_extrinsics_losses = []

    for i in pbar:
        if batch_size is None:
            batch = dataset
        else:
            bid = torch.randperm(len(dataset["mask"]))[:batch_size]
            # Only batch tensors that have a sample dimension (first dim == N)
            # intrinsic and mount_poses should not be batched
            batch = {}
            for k, v in dataset.items():
                if v is None:
                    batch[k] = None
                elif k in ["intrinsic"]:
                    # These should not be batched - same for all samples
                    batch[k] = v
                else:
                    # These should be batched - different per sample
                    batch[k] = v[bid]
        output = solver(batch)
        optimizer.zero_grad()
        output["mask_loss"].backward()
        loss_value = output["mask_loss"].item()

        # Check if gradients are non-zero (for debugging)
        grad_norm = 0.0
        grad_count = 0
        for param in solver.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                if torch.isfinite(torch.tensor(param_grad_norm)):
                    grad_norm += param_grad_norm**2
                    grad_count += 1

        if grad_count > 0:
            grad_norm = grad_norm**0.5
        else:
            grad_norm = 0.0

        # Debug info for first few iterations
        if i < 3 and verbose:
            total_params = sum(1 for _ in solver.parameters())
            if i == 0:
                print(
                    f"\nDebug - Total parameters: {total_params}, Parameters with gradients: {grad_count}"
                )
                if grad_count == 0:
                    print(
                        "WARNING: No parameters have gradients! Check if solver.parameters() returns trainable parameters."
                    )

            # Show detailed gradient info for first few iterations
            print(f"Iteration {i}: Loss = {output['mask_loss'].item():.6f}")
            print(f"  Output keys: {list(output.keys())}")

            # Debug the loss tensor properties
            loss_tensor = output["mask_loss"]
            print(
                f"  Loss tensor: requires_grad={loss_tensor.requires_grad}, grad_fn={loss_tensor.grad_fn}"
            )

            # Check if rendered masks are changing
            rendered_masks = output.get("rendered_masks", None)
            ref_masks = output.get("ref_masks", None)
            print(
                f"  Found rendered_masks: {rendered_masks is not None}, ref_masks: {ref_masks is not None}"
            )
            if rendered_masks is not None and ref_masks is not None:
                rendered_sum = rendered_masks.sum().item()
                ref_sum = ref_masks.sum().item()
                mse_loss = ((rendered_masks - ref_masks) ** 2).mean().item()

                # Calculate intersection and union for IoU
                rendered_binary = (rendered_masks > 0.5).float()
                ref_binary = (ref_masks > 0.5).float()
                intersection = (rendered_binary * ref_binary).sum().item()
                union = (rendered_binary + ref_binary).clamp(max=1).sum().item()
                iou = intersection / union if union > 0 else 0.0

                print(
                    f"  Masks: rendered_sum={rendered_sum:.2f}, ref_sum={ref_sum:.2f}, mse={mse_loss:.6f}"
                )
                print(
                    f"         intersection={intersection:.2f}, union={union:.2f}, iou={iou:.4f}"
                )

                if intersection == 0:
                    print(
                        "         *** NO OVERLAP BETWEEN MASKS - THIS EXPLAINS ZERO GRADIENTS ***"
                    )

                if i > 0 and prev_rendered_sum is not None:
                    change = abs(rendered_sum - prev_rendered_sum)
                    print(f"         rendered_mask_change_from_prev={change:.6f}")
                prev_rendered_sum = rendered_sum

            for j, param in enumerate(solver.parameters()):
                if param.grad is not None:
                    grad_norm_param = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    param_norm = param.norm().item()
                    print(
                        f"  Param {j}: shape={param.shape}, param_norm={param_norm:.6f}"
                    )
                    print(
                        f"           grad_norm={grad_norm_param:.6f}, grad_mean={grad_mean:.6f}, grad_std={grad_std:.6f}"
                    )
                    print(
                        f"           param_range=[{param.min().item():.6f}, {param.max().item():.6f}]"
                    )
                    if grad_norm_param == 0:
                        print("           *** ZERO GRADIENT DETECTED ***")
                        # Try to understand why gradient is zero
                        if hasattr(param.grad, "dtype"):
                            print(
                                f"           grad_dtype={param.grad.dtype}, requires_grad={param.requires_grad}"
                            )

                        # Manual gradient check on first iteration
                        if i == 0:
                            print("           Performing manual gradient check...")
                            with torch.no_grad():
                                original_param = param.clone()
                                eps_values = [
                                    1e-2,
                                    1e-3,
                                    1e-4,
                                    1e-5,
                                ]  # Try different step sizes

                                for eps in eps_values:
                                    # Forward pass with small perturbation
                                    param.data = original_param + eps
                                    output_plus = solver(batch)
                                    loss_plus = output_plus["mask_loss"].item()

                                    param.data = original_param - eps
                                    output_minus = solver(batch)
                                    loss_minus = output_minus["mask_loss"].item()

                                    # Compute numerical gradient
                                    numerical_grad = (loss_plus - loss_minus) / (
                                        2 * eps
                                    )
                                    print(
                                        f"           eps={eps:.1e}: grad={numerical_grad:.8f}, loss_plus={loss_plus:.2f}, loss_minus={loss_minus:.2f}"
                                    )

                                # Restore original parameter
                                param.data = original_param

                                # Check if the issue is in gradient computation vs loss sensitivity
                                print("           Testing larger perturbations...")
                                large_eps = 0.1
                                param.data = original_param + large_eps
                                output_large_plus = solver(batch)
                                loss_large_plus = output_large_plus["mask_loss"].item()

                                param.data = original_param - large_eps
                                output_large_minus = solver(batch)
                                loss_large_minus = output_large_minus[
                                    "mask_loss"
                                ].item()

                                param.data = original_param  # Restore

                                print(
                                    f"           Large step (eps={large_eps}): loss_plus={loss_large_plus:.2f}, loss_minus={loss_large_minus:.2f}"
                                )
                                if abs(loss_large_plus - loss_large_minus) < 1e-6:
                                    print(
                                        "           *** LOSS IS COMPLETELY FLAT - RENDERING ISSUE OR BAD INITIALIZATION ***"
                                    )
                else:
                    print(f"  Param {j}: No gradient")
            print()

        optimizer.step()
        all_losses.append(loss_value)
        if loss_value > worst_loss:
            worst_loss = loss_value
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
            average_loss = np.mean(all_losses)
            loss_std = np.std(all_losses) if len(all_losses) > 1 else 0.0
            grad_str = (
                f"{grad_norm:.2e}" if grad_count > 0 else f"0.0 ({grad_count} params)"
            )
            pbar.set_description(
                f"Loss: {loss_value:.2f}, Best: {best_loss:.2f}, Worst: {worst_loss:.2f}, "
                f"Avg: {average_loss:.2f}, Std: {loss_std:.4f}, Grad: {grad_str}"
            )
        if "metrics" in output:
            pbar.set_postfix(output["metrics"])
    if return_history:
        # return torch.stack(extrinsics)

        return {
            "best_extrinsics": torch.stack(best_extrinsics),
            "best_extrinsics_step": best_extrinsics_steps,
            "best_extrinsics_losses": best_extrinsics_losses,
            "all_losses": all_losses,
        }
    else:
        return best_predicted_extrinsic
