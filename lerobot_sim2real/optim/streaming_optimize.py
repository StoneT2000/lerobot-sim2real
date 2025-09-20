from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm

from easyhec.optim.rb_solver import RBSolver, RBSolverConfig


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
    learning_rate: float = 3e-3,
    batch_size: Optional[int] = None,
    early_stopping_steps: int = 1000,
    verbose: bool = True,
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
        else:
            bid = torch.randperm(len(dataset["mask"]))[:batch_size]
            batch = {
                k: v[bid] if hasattr(v, "__getitem__") else v
                for k, v in dataset.items()
            }

        output = solver(batch)
        optimizer.zero_grad()
        output["mask_loss"].backward()
        optimizer.step()

        loss_value = float(output["mask_loss"].item())
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
