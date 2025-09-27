"""Run optimize_streaming from saved results and visualize initial vs optimized.

Loads masks, intrinsics, link poses, and images from results/<uid>/<robot_id>/<camera>/,
rebuilds meshes from the URDF, runs optimize_streaming, and saves visualizations
using custom_visualization.visualize_extrinsic_results_red_mask.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro
from transforms3d.euler import euler2mat
from easyhec.utils.camera_conversions import ros2opencv

# Import shared URDF utilities
from lerobot_sim2real.utils.urdf_utils import load_robot_meshes_for_calibration

from lerobot_sim2real.optim.streaming_optimize import optimize_streaming
from lerobot_sim2real.utils.camera import scale_intrinsics  # noqa: F401 (kept for quick toggling)
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer

# Import visualization helper from project root
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from custom_visualization import visualize_extrinsic_results_red_mask


@dataclass
class Args:
    # Base directory like results/so101/so101_follower
    results_root: str = "results/so101/so101_follower"
    # Camera subdir under results_root
    camera_name: str = "base_camera"
    # Robot UID: so100 or so101 (used to choose URDF path)
    uid: str = "so101"

    # Optimization hyperparameters
    iterations: int = 5000
    learning_rate: float = 3e-3
    early_stopping_steps: int = 200
    verbose: bool = True

    # Output directory for visualization
    output_subdir: str = "debug_optimize"

    # Where to get intrinsics: 'file' or 'hardcoded' (alias: 'live' -> hardcoded)
    intrinsics_source: str = "file"

    # Intrinsics tweak knobs
    fx_scale: float = 1.0
    fy_scale: float = 1.0
    cx_offset_px: float = 0.0
    cy_offset_px: float = 0.0

    # Renderer/extrinsic conventions
    invert_extrinsic: bool = True
    swap_multiplication_order: bool = False
    auto_tune_conventions: bool = True

    # Initial extrinsic options
    initial_extrinsic_path: Optional[str] = None
    use_saved_initial_if_available: bool = True
    init_rx_deg: float = 0.0
    init_ry_deg: float = 45.0
    init_rz_deg: float = -36.0
    init_tx: float = -0.4
    init_ty: float = 0.1
    init_tz: float = 0.5

    # Coarse alignment to intersect the mask before optimization
    coarse_align_initial: bool = True
    coarse_align_max_iters: int = 100
    coarse_align_initial_step: float = 0.05


def build_meshes_from_urdf(uid: str) -> List:
    """Build meshes from URDF using shared utility function."""
    _, meshes, _ = load_robot_meshes_for_calibration(uid)
    return meshes


def compute_initial_guess_opencv_from_args(args: Args) -> np.ndarray:
    # Build from CLI-provided rotation (degrees) and translation
    rx = np.deg2rad(args.init_rx_deg)
    ry = np.deg2rad(args.init_ry_deg)
    rz = np.deg2rad(args.init_rz_deg)
    initial_extrinsic_guess = np.eye(4, dtype=np.float32)
    initial_extrinsic_guess[:3, :3] = euler2mat(rx, ry, rz)
    initial_extrinsic_guess[:3, 3] = np.array(
        [args.init_tx, args.init_ty, args.init_tz], dtype=np.float32
    )
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)
    return initial_extrinsic_guess


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.results_root)
    cam_dir = root / args.camera_name

    # Load datasets
    link_poses_dataset = np.load(root / "link_poses_dataset.npy")
    # image_dataset.npy is a dict (camera_name -> images)
    image_dataset_obj = np.load(root / "image_dataset.npy", allow_pickle=True).reshape(
        -1
    )[0]
    images = image_dataset_obj[args.camera_name]

    masks = np.load(cam_dir / "mask.npy")
    # Dataset dimensions from masks
    camera_height, camera_width = masks.shape[1], masks.shape[2]

    # Prefer saved intrinsics if available
    intrinsic_path = cam_dir / "camera_intrinsic.npy"
    if args.intrinsics_source == "file" and intrinsic_path.exists():
        intrinsic = np.load(intrinsic_path)
    else:
        # Use the same hard-coded base K (for 640x480) and scale to dataset resolution
        K_base = np.array(
            [[595.9, 0.0, 320.0], [0.0, 794.7, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        if (camera_width, camera_height) == (640, 480):
            intrinsic = K_base
        else:
            # Scale to dataset resolution if different
            from lerobot_sim2real.utils.camera import scale_intrinsics as _scale

            intrinsic = _scale(K_base, 640, 480, camera_width, camera_height)

    # Apply user tweaks to intrinsic matrix
    intrinsic = intrinsic.copy().astype(np.float32)
    intrinsic[0, 0] *= float(args.fx_scale)
    intrinsic[1, 1] *= float(args.fy_scale)
    intrinsic[0, 2] += float(args.cx_offset_px)
    intrinsic[1, 2] += float(args.cy_offset_px)

    # Initial guess
    initial_extrinsic_guess: Optional[np.ndarray] = None
    # Prefer explicit path
    if (
        args.initial_extrinsic_path is not None
        and Path(args.initial_extrinsic_path).exists()
    ):
        initial_extrinsic_guess = np.load(args.initial_extrinsic_path)
    # Then previously saved optimized extrinsic in camera folder
    elif (
        args.use_saved_initial_if_available
        and (cam_dir / "camera_extrinsic_opencv.npy").exists()
    ):
        initial_extrinsic_guess = np.load(cam_dir / "camera_extrinsic_opencv.npy")
    # Else compute from CLI-provided defaults
    else:
        initial_extrinsic_guess = compute_initial_guess_opencv_from_args(args)

    # Build meshes from URDF
    meshes = build_meshes_from_urdf(args.uid)

    # Prepare renderer tensors for coarse alignment and later visualization
    renderer = NVDiffrastRenderer(camera_height, camera_width)
    link_vertices = [
        torch.from_numpy(mesh.vertices.copy()).float().to(device) for mesh in meshes
    ]
    link_faces = [
        torch.from_numpy(mesh.faces.copy()).int().to(device) for mesh in meshes
    ]
    intrinsic_t = torch.from_numpy(intrinsic).float().to(device)
    link_poses_all_t = torch.from_numpy(link_poses_dataset).float().to(device)

    # Utility to render with current/overridden conventions
    def render_mask_from_extrinsic(
        extr_np: np.ndarray,
        img_i: int,
        invert: Optional[bool] = None,
        swap: Optional[bool] = None,
    ) -> np.ndarray:
        mask_pred = torch.zeros((camera_height, camera_width), device=device)
        extr_t = torch.from_numpy(extr_np).float().to(device)
        use_invert = args.invert_extrinsic if invert is None else invert
        use_swap = args.swap_multiplication_order if swap is None else swap
        cam_pose = torch.linalg.inv(extr_t) if use_invert else extr_t
        link_poses_t = link_poses_all_t[img_i]
        for j in range(len(link_vertices)):
            composed = (
                (cam_pose @ link_poses_t[j])
                if not use_swap
                else (link_poses_t[j] @ cam_pose)
            )
            link_mask = renderer.render_mask(
                link_vertices[j], link_faces[j], intrinsic_t, composed
            )
            mask_pred[link_mask > 0] = 1
        return mask_pred.detach().cpu().numpy()

    # Auto-tune invert/swap by IoU on first frame
    if args.auto_tune_conventions:
        combos = [(True, False), (False, False), (True, True), (False, True)]
        target = (masks[0] > 0.5).astype(bool)
        best_score = -1.0
        best_combo = (args.invert_extrinsic, args.swap_multiplication_order)
        for inv, swp in combos:
            pred = render_mask_from_extrinsic(
                initial_extrinsic_guess, 0, invert=inv, swap=swp
            )
            pred_b = pred.astype(bool)
            inter = (pred_b & target).sum()
            union = (pred_b | target).sum()
            score = float(inter) / float(union) if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_combo = (inv, swp)
        args.invert_extrinsic, args.swap_multiplication_order = best_combo

    # Optionally coarse-align translation to intersect mask on first image
    if args.coarse_align_initial:

        def iou(a: np.ndarray, b: np.ndarray) -> float:
            a_bool = a.astype(bool)
            b_bool = b.astype(bool)
            inter = np.logical_and(a_bool, b_bool).sum()
            union = np.logical_or(a_bool, b_bool).sum()
            return float(inter) / float(union) if union > 0 else 0.0

        best = initial_extrinsic_guess.copy()
        best_score = iou(render_mask_from_extrinsic(best, 0), masks[0] > 0.5)
        step = args.coarse_align_initial_step
        for _ in range(args.coarse_align_max_iters):
            improved = False
            for dx, dy, dz in [
                (step, 0.0, 0.0),
                (-step, 0.0, 0.0),
                (0.0, step, 0.0),
                (0.0, -step, 0.0),
                (0.0, 0.0, step),
                (0.0, 0.0, -step),
            ]:
                cand = best.copy()
                cand[:3, 3] += np.array([dx, dy, dz], dtype=np.float32)
                score = iou(render_mask_from_extrinsic(cand, 0), masks[0] > 0.5)
                if score > best_score:
                    best = cand
                    best_score = score
                    improved = True
            if not improved:
                step *= 0.5
                if step < 0.005:
                    break
        initial_extrinsic_guess = best

    # Torch tensors
    init_extr_t = torch.from_numpy(initial_extrinsic_guess).float().to(device)
    masks_t = torch.from_numpy(masks).float().to(device)
    link_poses_t = link_poses_all_t

    # Run optimization
    hist = optimize_streaming(
        initial_extrinsic_guess=init_extr_t,
        camera_intrinsic=intrinsic_t,
        masks=masks_t,
        link_poses_dataset=link_poses_t,
        meshes=meshes,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_mount_poses=None,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        early_stopping_steps=args.early_stopping_steps,
        verbose=args.verbose,
    )

    final_best = hist["final_best_extrinsic"].detach().cpu().numpy()

    # Visualize initial vs optimized
    out_dir = cam_dir / args.output_subdir
    visualize_extrinsic_results_red_mask(
        images=images,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        intrinsic=intrinsic,
        extrinsics=np.stack([initial_extrinsic_guess, final_best]),
        masks=masks,
        labels=["Initial Extrinsic Guess", "Optimized Extrinsic"],
        output_dir=str(out_dir),
        mask_color=(255, 0, 0),
        invert_extrinsic=args.invert_extrinsic,
        swap_multiplication_order=args.swap_multiplication_order,
    )

    # Save optimized extrinsic for reuse
    np.save(out_dir / "camera_extrinsic_opencv_debug.npy", final_best)
    print(f"Saved visualization to: {out_dir}")
    print(
        f"Saved optimized extrinsic to: {out_dir / 'camera_extrinsic_opencv_debug.npy'}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
