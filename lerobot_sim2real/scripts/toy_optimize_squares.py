"""Toy example: align two 2D squares by optimizing center translation.

We create a soft (differentiable) square mask parameterized by its center (cx, cy)
and side length s. Given a fixed target mask, we optimize (cx, cy) (and optionally s)
to maximize overlap (minimize MSE + soft IoU loss).

Usage:
  python -m lerobot_sim2real.scripts.toy_optimize_squares \
    --width 128 --height 128 --target_cx 80 --target_cy 64 --side 40 \
    --init_cx 20 --init_cy 20 --optimize_side False --steps 1000 --lr 5e-2

Outputs:
  - Saves overlays of target vs predicted at start and end into --out_dir.
  - Prints stepwise loss and IoU progress if --verbose.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import cv2


@dataclass
class Args:
    width: int = 128
    height: int = 128
    # Target square params (in pixel coordinates)
    target_cx: float = 80.0
    target_cy: float = 64.0
    side: float = 40.0

    # Initial guess for predicted square
    init_cx: float = 20.0
    init_cy: float = 20.0
    init_side: float = 40.0
    optimize_side: bool = False

    # Optimization
    steps: int = 1000
    lr: float = 5e-2
    iou_weight: float = 0.2
    com_weight: float = 0.8  # center-of-mass alignment weight
    edge_steepness: float = 0.25  # Larger -> sharper rectangle edges
    verbose: bool = True

    out_dir: str = "results/toy_squares"


def build_coord_grid(
    height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(0, height, device=device).float().view(-1, 1).repeat(1, width)
    xs = torch.arange(0, width, device=device).float().view(1, -1).repeat(height, 1)
    return xs, ys


def soft_square_mask(
    xs: torch.Tensor,
    ys: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    side_len: torch.Tensor,
    k: float,
) -> torch.Tensor:
    half = side_len * 0.5
    left = center_x - half
    right = center_x + half
    top = center_y - half
    bottom = center_y + half

    # Sigmoid edges to keep gradients
    sx1 = torch.sigmoid(k * (xs - left))
    sx2 = torch.sigmoid(k * (right - xs))
    sy1 = torch.sigmoid(k * (ys - top))
    sy2 = torch.sigmoid(k * (bottom - ys))
    mask = sx1 * sx2 * sy1 * sy2
    return mask


def tensor_to_image(mask: torch.Tensor) -> np.ndarray:
    m = mask.detach().cpu().clamp(0, 1).numpy()
    m_u8 = (m * 255).astype(np.uint8)
    return m_u8


def overlay_masks(target: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    t = tensor_to_image(target)
    p = tensor_to_image(pred)
    h, w = t.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # target in green edges
    edges_t = cv2.Canny(t, 50, 150)
    rgb[edges_t > 0] = (0, 255, 0)
    # predicted in red fill
    rgb[p > 127] = (255, 0, 0)
    return rgb


def compute_soft_iou(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    inter = (pred * ref).sum()
    union = (pred + ref - pred * ref).sum()
    return (inter + eps) / (union + eps)


def center_of_mass(
    mask: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-6
    m = mask.clamp(min=0.0)
    mass = m.sum() + eps
    cx = (m * xs).sum() / mass
    cy = (m * ys).sum() / mass
    return cx, cy


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.height, args.width
    xs, ys = build_coord_grid(H, W, device)

    # Build target mask
    target = soft_square_mask(
        xs,
        ys,
        torch.tensor(args.target_cx, device=device),
        torch.tensor(args.target_cy, device=device),
        torch.tensor(args.side, device=device),
        k=float(args.edge_steepness),
    )

    # Parameters to optimize (unconstrained)
    cx = torch.nn.Parameter(torch.tensor(args.init_cx, device=device).float())
    cy = torch.nn.Parameter(torch.tensor(args.init_cy, device=device).float())
    if args.optimize_side:
        side = torch.nn.Parameter(torch.tensor(args.init_side, device=device).float())
        params = [cx, cy, side]
    else:
        side = torch.tensor(args.init_side, device=device).float()
        params = [cx, cy]

    optim = torch.optim.Adam(params, lr=float(args.lr))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pred0 = soft_square_mask(xs, ys, cx, cy, side, k=float(args.edge_steepness))
        cv2.imwrite(str(out_dir / "start.png"), overlay_masks(target, pred0))
        tgt_cx, tgt_cy = center_of_mass(target, xs, ys)

    best_iou = -1.0
    for step in range(int(args.steps)):
        pred = soft_square_mask(xs, ys, cx, cy, side, k=float(args.edge_steepness))
        pred_smooth = (
            F.avg_pool2d(pred.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze(0).squeeze(0)
        )

        mse = F.mse_loss(pred_smooth, target)
        siou = compute_soft_iou(pred_smooth, target)
        # Center-of-mass alignment term (pixels)
        pred_cx, pred_cy = center_of_mass(pred_smooth, xs, ys)
        com_l2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2
        # Combine losses
        alpha = float(args.iou_weight)
        beta = float(args.com_weight)
        w = max(0.0, 1.0 - alpha - beta)
        loss = w * mse + alpha * (1.0 - siou) + beta * (com_l2 / (W**2 + H**2))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        if args.verbose and (step % 50 == 0 or step == args.steps - 1):
            print(
                f"step={step:04d} loss={float(loss.item()):.6f} "
                f"mse={float(mse.item()):.6f} iou={float(siou.item()):.4f} "
                f"cx={float(cx.item()):.2f} cy={float(cy.item()):.2f}"
                + (f" s={float(side.item()):.2f}" if args.optimize_side else "")
            )
        best_iou = max(best_iou, float(siou.item()))

    with torch.no_grad():
        predF = soft_square_mask(xs, ys, cx, cy, side, k=float(args.edge_steepness))
        cv2.imwrite(str(out_dir / "final.png"), overlay_masks(target, predF))
        print(f"Saved overlays to: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
