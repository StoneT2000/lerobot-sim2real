from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm

from .rb_solver_override import RBSolver, RBSolverConfig  # type: ignore

# from easyhec.optim.rb_solver import RBSolver, RBSolverConfig  # type: ignore
import torch.nn.functional as F


@torch.no_grad()
def _clone_extrinsic(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone()


def optimize_streaming(
    initial_extrinsic_guess: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    masks: torch.Tensor,
    link_poses_dataset: torch.Tensor,
    meshes: List,
    camera_width: int,
    camera_height: int,
    camera_mount_poses: Optional[torch.Tensor] = None,
    iterations: int = 5000,
    learning_rate: float = 1e-3,
    batch_size: Optional[int] = 1,
    early_stopping_steps: int = 300,
    verbose: bool = True,
    loss_multiplier: float = 1.0,
    # Stabilization options
    iou_loss_weight: float = 0.1,
    grad_clip_norm: float = 1.0,
    freeze_rotation_steps: int = 50,
    warmup_first_sample_steps: int = 200,
    # Two-phase LR schedule
    lr_phase1: float = 2e-3,
    lr_phase2: float = 5e-4,
    phase1_steps: int = 200,
    # Center-of-mass alignment weight (0 to disable)
    com_loss_weight: float = 0.2,
) -> Dict[str, Any]:
    """
    Variant of easyhec.optim.optimize that returns per-step histories for visualization.

    Returns a dict with:
      - losses: List[float] length T
      - current_extrinsics: torch.Tensor [T, 4, 4] (solver prediction each step)
      - best_so_far_extrinsics: torch.Tensor [T, 4, 4] (best solution up to each step)
      - final_best_extrinsic: torch.Tensor [4, 4]
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
    solver = RBSolver(cfg).to(device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=learning_rate)

    best_predicted_extrinsic = initial_extrinsic_guess.detach().clone()
    best_loss = float("inf")
    last_loss_improvement_step = 0

    pbar = tqdm(range(iterations)) if verbose else range(iterations)
    dataset = dict(
        intrinsic=camera_intrinsic,
        link_poses=link_poses_dataset,
        mask=masks,
        mount_poses=camera_mount_poses,
    )

    losses: List[float] = []
    current_list: List[torch.Tensor] = []
    best_so_far_list: List[torch.Tensor] = []

    for i in pbar:
        if batch_size is None:
            batch = dataset
            bid = None
        else:
            # Build a safe batch: only index tensors that are sample-aligned (first dim == N)
            N = len(dataset["mask"])  # number of samples
            if i < warmup_first_sample_steps:
                bid = torch.tensor([0])
            else:
                bid = torch.randperm(N)[:batch_size]

            def maybe_index(name: str, value):
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() >= 1
                    and value.shape[0] == N
                ):
                    return value[bid]
                return value

            batch = {
                "intrinsic": dataset["intrinsic"],
                "link_poses": maybe_index("link_poses", dataset["link_poses"]),
                "mask": maybe_index("mask", dataset["mask"]),
                "mount_poses": maybe_index("mount_poses", dataset.get("mount_poses")),
            }

        # Simple 2-phase LR schedule
        if i == 0:
            for g in optimizer.param_groups:
                g["lr"] = lr_phase1
        elif i == phase1_steps:
            for g in optimizer.param_groups:
                g["lr"] = lr_phase2

        output = solver(batch)
        optimizer.zero_grad()

        # Mild smoothing on prediction to improve gradient signal
        pred_raw = output["rendered_masks"].float()
        pred = F.avg_pool2d(
            pred_raw.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        ref = output["ref_masks"].float()

        # Recompute MSE with smoothed pred
        mse_loss = F.mse_loss(pred, ref)
        total_loss = mse_loss * loss_multiplier
        if iou_loss_weight > 0:
            # Blend with soft IoU (Dice/IoU-like) to reduce shrinking/expanding degeneracies
            eps = 1e-6
            inter = (pred * ref).sum(dim=(1, 2))
            union = (pred + ref - pred * ref).sum(dim=(1, 2))
            soft_iou = (inter + eps) / (union + eps)
            iou_loss = 1.0 - soft_iou.mean()
            total_loss = (
                1.0 - iou_loss_weight
            ) * total_loss + iou_loss_weight * iou_loss

        # Center-of-mass alignment to improve large-offset gradients
        if com_loss_weight > 0:
            B, H, W = pred.shape
            device_b = pred.device
            ys = torch.arange(0, H, device=device_b).float().view(-1, 1).repeat(1, W)
            xs = torch.arange(0, W, device=device_b).float().view(1, -1).repeat(H, 1)
            xs = xs.unsqueeze(0).expand(B, -1, -1)
            ys = ys.unsqueeze(0).expand(B, -1, -1)
            eps = 1e-6
            pred_m = pred.clamp(min=0.0)
            ref_m = ref.clamp(min=0.0)
            pred_mass = pred_m.sum(dim=(1, 2)) + eps
            ref_mass = ref_m.sum(dim=(1, 2)) + eps
            pred_cx = (pred_m * xs).sum(dim=(1, 2)) / pred_mass
            pred_cy = (pred_m * ys).sum(dim=(1, 2)) / pred_mass
            ref_cx = (ref_m * xs).sum(dim=(1, 2)) / ref_mass
            ref_cy = (ref_m * ys).sum(dim=(1, 2)) / ref_mass
            com_l2 = ((pred_cx - ref_cx) ** 2 + (pred_cy - ref_cy) ** 2) / (W**2 + H**2)
            com_loss = com_l2.mean()
            total_loss = (
                1.0 - com_loss_weight
            ) * total_loss + com_loss_weight * com_loss

        total_loss.backward()

        # Optionally freeze rotation (last 3 dof) for initial steps to prevent early drift
        if (
            i < freeze_rotation_steps
            and hasattr(solver, "dof")
            and solver.dof.grad is not None
        ):
            solver.dof.grad[3:] = 0.0

        # Gradient clipping for stability
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(solver.parameters(), grad_clip_norm)
        optimizer.step()

        loss_value = float(total_loss.item())
        losses.append(loss_value)

        current_pred = solver.get_predicted_extrinsic()
        current_list.append(_clone_extrinsic(current_pred))

        if loss_value < best_loss:
            best_loss = loss_value
            best_predicted_extrinsic = _clone_extrinsic(current_pred)
            last_loss_improvement_step = i

        best_so_far_list.append(_clone_extrinsic(best_predicted_extrinsic))

        if i - last_loss_improvement_step >= early_stopping_steps:
            break
        if verbose:
            pbar.set_description(f"Loss: {loss_value:.2f}, Best Loss: {best_loss:.2f}")
        if "metrics" in output:
            pbar.set_postfix(output["metrics"])  # type: ignore[arg-type]

    return {
        "losses": losses,
        "current_extrinsics": torch.stack(current_list)
        if len(current_list) > 0
        else torch.empty((0, 4, 4), device=device),
        "best_so_far_extrinsics": torch.stack(best_so_far_list)
        if len(best_so_far_list) > 0
        else torch.empty((0, 4, 4), device=device),
        "final_best_extrinsic": best_predicted_extrinsic,
    }


@torch.no_grad()
def optimize_final(
    initial_extrinsic_guess: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    masks: torch.Tensor,
    link_poses_dataset: torch.Tensor,
    meshes: List,
    camera_width: int,
    camera_height: int,
    camera_mount_poses: Optional[torch.Tensor] = None,
    iterations: int = 5000,
    learning_rate: float = 3e-4,
    batch_size: Optional[int] = None,
    early_stopping_steps: int = 1000,
    verbose: bool = False,
) -> torch.Tensor:
    hist = optimize_streaming(
        initial_extrinsic_guess=initial_extrinsic_guess,
        camera_intrinsic=camera_intrinsic,
        masks=masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_mount_poses=camera_mount_poses,
        iterations=iterations,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping_steps=early_stopping_steps,
        verbose=verbose,
    )
    return hist["final_best_extrinsic"].detach().clone()
