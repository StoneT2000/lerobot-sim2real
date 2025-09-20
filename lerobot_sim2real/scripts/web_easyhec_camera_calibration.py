"""Web-based EasyHEC camera calibration with Gradio mask UI.

Reference for the original interactive segmentation flow:
https://github.com/StoneT2000/simple-easyhec/blob/main/easyhec/segmentation/interactive.py
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import tyro
import gradio as gr
from lerobot.cameras.realsense import RealSenseCamera

from lerobot.motors.motors_bus import MotorNormMode
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from transforms3d.euler import euler2mat
from urchin import URDF

from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from easyhec.utils.utils_3d import merge_meshes
import cv2

# Optional: red mask overlay visualization
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from custom_visualization import visualize_extrinsic_results_red_mask
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer

from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.utils.camera import scale_intrinsics
from lerobot_sim2real.optim.streaming_optimize import optimize_streaming


def _k_from_fov(
    width: int, height: int, hfov_deg: Optional[float], vfov_deg: Optional[float]
):
    import math

    cx = width / 2.0
    cy = height / 2.0
    fx = None
    fy = None
    if hfov_deg is not None:
        fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    if vfov_deg is not None:
        fy = 0.5 * height / math.tan(math.radians(vfov_deg) / 2.0)
    if fx is not None and fy is None:
        fy = fx * (width / height)
    if fy is not None and fx is None:
        fx = fy * (height / width)
    if fx is None or fy is None:
        return None
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _normalize_sam2_cfg_name(cfg: str) -> str:
    s = str(cfg)
    if "configs/" in s:
        return s[s.index("configs/") :]
    if "/" in s:
        base = s
        if not (base.endswith(".yaml") or base.endswith(".yml")):
            base = f"{base}.yaml"
        return f"configs/{base}"
    name = s
    if not (name.endswith(".yaml") or name.endswith(".yml")):
        name = f"{name}.yaml"
    return f"configs/sam2.1/{name}"


class WebMaskAnnotator:
    """Gradio UI for point-and-click mask creation using SAM2.

    - Click on the image to add points. Use label radio to switch pos/neg.
    - Generate mask to preview. Accept to store per-image mask.
    - Navigate with Prev/Next. Finish & Save writes masks and signals completion.
    """

    def __init__(
        self,
        images: np.ndarray,
        model_cfg: str,
        checkpoint: str,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        mask_path: Optional[Path] = None,
        # Optional optimization context for live preview
        intrinsic: Optional[np.ndarray] = None,
        link_poses_dataset: Optional[np.ndarray] = None,
        meshes: Optional[List] = None,
        initial_extrinsic_guess: Optional[np.ndarray] = None,
        camera_mount_poses: Optional[np.ndarray] = None,
        iterations: int = 5000,
        early_stopping_steps: int = 200,
    ) -> None:
        self.images = images
        self.num_images = len(images)
        self.mask_path = mask_path
        self.done_event = threading.Event()
        self.server_name = server_name
        self.server_port = server_port
        # Live preview state
        self.optim_intrinsic = intrinsic
        self.optim_link_poses_dataset = link_poses_dataset
        self.optim_meshes = meshes
        self.optim_initial_extrinsic_guess = initial_extrinsic_guess
        self.optim_camera_mount_poses = camera_mount_poses
        self.optim_iterations = iterations
        self.optim_early_stopping = early_stopping_steps
        self._renderer = None
        self._link_vertices = None
        self._link_faces = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build predictor once (headless-friendly)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(
            build_sam2(
                _normalize_sam2_cfg_name(model_cfg), checkpoint, device=device_str
            )
        )

        # Per-image state
        self.clicked_points: List[List[Tuple[int, int, int]]] = [
            list() for _ in range(self.num_images)
        ]
        self.masks: List[Optional[np.ndarray]] = [None for _ in range(self.num_images)]

        # Build UI
        self.app = self._build_ui()

    @staticmethod
    def _overlay_mask(
        image: np.ndarray, mask: Optional[np.ndarray], color=(255, 0, 0), alpha=0.6
    ) -> np.ndarray:
        if mask is None:
            return image
        mask = mask.astype(bool)
        result = image.copy()
        overlay = np.full_like(result, color)
        result[mask] = ((1 - alpha) * result[mask] + alpha * overlay[mask]).astype(
            np.uint8
        )
        return result

    def _build_ui(self) -> gr.Blocks:
        with gr.Blocks() as demo:
            gr.Markdown("# Effortless Simple EasyHEC")
            with gr.Row():
                with gr.Column(scale=3):
                    image_view = gr.Image(
                        value=self.images[0],
                        label="Image",
                        interactive=True,
                        type="numpy",
                    )
                with gr.Column():
                    idx = gr.Number(value=0, label="Image Index", precision=0)
                    label_radio = gr.Radio(
                        [1, -1], value=1, label="Point Label (1=pos, -1=neg)"
                    )
                    status = gr.Textbox(label="Status", interactive=False)
                    btn_prev = gr.Button("Prev")
                    btn_next = gr.Button("Next")
                    btn_clear = gr.Button("Clear Points")
                    btn_generate = gr.Button("Generate Mask")
                    btn_accept = gr.Button("Accept Mask")
                    btn_save = gr.Button("Save Masks")
                    btn_exit = gr.Button("Finish & Exit")

            # Live optimization preview section
            gr.Markdown("## Live Optimization Preview (best-improvement trajectory)")
            with gr.Row():
                with gr.Column(scale=3):
                    opt_image = gr.Image(label="Overlay (Current Best Extrinsic)")
                with gr.Column():
                    opt_idx = gr.Number(
                        value=0, label="Preview Image Index", precision=0
                    )
                    opt_info = gr.Textbox(label="Progress", interactive=False)
                    btn_preview = gr.Button("Start Preview")
            with gr.Row():
                init_xyz = gr.Markdown("")
            with gr.Row():
                step_table = gr.Dataframe(
                    headers=["step", "loss", "best_loss", "tx", "ty", "tz"],
                    interactive=False,
                    wrap=True,
                    label="Prediction Steps",
                )

            def load_image(i: float) -> np.ndarray:
                i = int(i)
                base = self.images[i]
                return base

            def on_select(evt: gr.SelectData, i: float, lbl: int) -> str:
                i = int(i)
                x, y = evt.index[0], evt.index[1]
                self.clicked_points[i].append((int(x), int(y), int(lbl)))
                return f"Image {i}: {len(self.clicked_points[i])} points"

            def redraw(i: float) -> np.ndarray:
                i = int(i)
                # If a mask exists for this image, overlay it; else just return base
                return WebMaskAnnotator._overlay_mask(self.images[i], self.masks[i])

            def clear_points(i: float) -> Tuple[np.ndarray, str]:
                i = int(i)
                self.clicked_points[i] = []
                self.masks[i] = None
                return load_image(i), f"Image {i}: cleared points"

            def prev(i: float) -> Tuple[float, np.ndarray, str]:
                i = int(i)
                i = max(0, i - 1)
                return float(i), redraw(i), f"Image {i}: loaded"

            def next_(i: float) -> Tuple[float, np.ndarray, str]:
                i = int(i)
                i = min(self.num_images - 1, i + 1)
                return float(i), redraw(i), f"Image {i}: loaded"

            def generate_mask(i: float) -> Tuple[np.ndarray, str]:
                i = int(i)
                pts = self.clicked_points[i]
                if len(pts) == 0:
                    return load_image(
                        i
                    ), "Add at least one point before generating mask"
                input_label = np.array([p[2] for p in pts], dtype=np.int32)
                input_point = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
                with torch.inference_mode():
                    self.predictor.set_image(self.images[i])
                    mask, _, _ = self.predictor.predict(
                        input_point, input_label, multimask_output=False
                    )
                self.masks[i] = mask[0]
                return WebMaskAnnotator._overlay_mask(
                    self.images[i], self.masks[i]
                ), f"Image {i}: mask generated"

            def accept_mask(i: float) -> Tuple[np.ndarray, str]:
                i = int(i)
                if self.masks[i] is None:
                    return load_image(i), "No mask to accept. Generate first."
                return redraw(
                    i
                ), f"Image {i}: mask accepted ({int(self.masks[i].sum())} px)"

            def save_masks() -> str:
                # Ensure all masks exist; if any missing, fill with zeros shape(H,W)
                H, W = self.images[0].shape[:2]
                final_masks = []
                for i in range(self.num_images):
                    if self.masks[i] is None:
                        final_masks.append(np.zeros((H, W), dtype=np.float32))
                    else:
                        final_masks.append(self.masks[i].astype(np.float32))
                masks_np = np.stack(final_masks)
                if self.mask_path is not None:
                    self.mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.mask_path, masks_np)
                return f"Saved masks to {self.mask_path if self.mask_path is not None else '(not saved)'}"

            def finish_and_exit() -> str:
                self.done_event.set()
                return "Exiting and continuing calibration..."

            # --- Live preview helpers ---
            def _ensure_renderer_init():
                if self._renderer is not None:
                    return
                if (
                    self.optim_intrinsic is None
                    or self.optim_link_poses_dataset is None
                    or self.optim_meshes is None
                    or self.optim_initial_extrinsic_guess is None
                ):
                    raise gr.Error("Optimization context not available yet.")
                H, W = self.images[0].shape[:2]
                self._renderer = NVDiffrastRenderer(H, W)
                # Prepare tensors
                self._link_vertices = [
                    torch.from_numpy(mesh.vertices.copy()).float().to(self._device)
                    for mesh in self.optim_meshes
                ]
                self._link_faces = [
                    torch.from_numpy(mesh.faces.copy()).int().to(self._device)
                    for mesh in self.optim_meshes
                ]

            def _render_mask_from_extrinsic(
                extr_np: np.ndarray, img_i: int
            ) -> np.ndarray:
                _ensure_renderer_init()
                H, W = self.images[0].shape[:2]
                mask = torch.zeros((H, W), device=self._device)
                intrinsic_t = (
                    torch.from_numpy(self.optim_intrinsic).float().to(self._device)
                )
                link_poses_t = (
                    torch.from_numpy(self.optim_link_poses_dataset[img_i])
                    .float()
                    .to(self._device)
                )
                extrinsic_t = torch.from_numpy(extr_np).float().to(self._device)
                if self.optim_camera_mount_poses is not None:
                    mount_pose = (
                        torch.from_numpy(self.optim_camera_mount_poses[img_i])
                        .float()
                        .to(self._device)
                    )
                    extrinsic_t = extrinsic_t @ mount_pose
                for j in range(len(self._link_vertices)):
                    link_mask = self._renderer.render_mask(
                        self._link_vertices[j],
                        self._link_faces[j],
                        intrinsic_t,
                        extrinsic_t @ link_poses_t[j],
                    )
                    mask[link_mask > 0] = 1
                return mask.detach().cpu().numpy()

            def _overlay(
                image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5
            ):
                m = mask.astype(bool)
                out = image.copy()
                if m.any():
                    overlay = np.full_like(out, color)
                    out[m] = ((1 - alpha) * out[m] + alpha * overlay[m]).astype(
                        np.uint8
                    )
                return out

            def start_preview(preview_idx: float):
                i_img = int(preview_idx)
                # Call optimizer in history mode to get best-improvement trajectory
                if (
                    self.optim_intrinsic is None
                    or self.optim_link_poses_dataset is None
                    or self.optim_initial_extrinsic_guess is None
                    or self.mask_path is None
                ):
                    raise gr.Error(
                        "Missing masks or optimization context. Save masks first."
                    )
                _ensure_renderer_init()

                # Build tensors for streaming
                intrinsic_t = (
                    torch.from_numpy(self.optim_intrinsic).float().to(self._device)
                )
                link_poses_t = (
                    torch.from_numpy(self.optim_link_poses_dataset)
                    .float()
                    .to(self._device)
                )
                masks_np = np.load(self.mask_path)
                masks_t = torch.from_numpy(masks_np).float().to(self._device)
                init_extr_t = (
                    torch.from_numpy(self.optim_initial_extrinsic_guess)
                    .float()
                    .to(self._device)
                )
                mount_poses_t = (
                    torch.from_numpy(self.optim_camera_mount_poses)
                    .float()
                    .to(self._device)
                    if self.optim_camera_mount_poses is not None
                    else None
                )

                hist = optimize_streaming(
                    initial_extrinsic_guess=init_extr_t,
                    camera_intrinsic=intrinsic_t,
                    masks=masks_t,
                    link_poses_dataset=link_poses_t,
                    meshes=self.optim_meshes,
                    camera_width=self.images.shape[2],
                    camera_height=self.images.shape[1],
                    camera_mount_poses=mount_poses_t,
                    iterations=self.optim_iterations,
                    early_stopping_steps=self.optim_early_stopping,
                    verbose=True,
                )

                best_hist = hist["best_so_far_extrinsics"].detach().cpu().numpy()
                losses = hist["losses"]
                init_tx, init_ty, init_tz = self.optim_initial_extrinsic_guess[
                    :3, 3
                ].tolist()
                init_text = f"**Initial translation**: tx={init_tx:.4f}, ty={init_ty:.4f}, tz={init_tz:.4f}"
                if best_hist.shape[0] == 0:
                    mask_prev = _render_mask_from_extrinsic(
                        self.optim_initial_extrinsic_guess, i_img
                    )
                    rows = [
                        [
                            0,
                            losses[0] if len(losses) else None,
                            losses[0] if len(losses) else None,
                            *self.optim_initial_extrinsic_guess[:3, 3].tolist(),
                        ]
                    ]
                    yield (
                        _overlay(self.images[i_img], mask_prev),
                        "no improvement",
                        0,
                        init_text,
                        rows,
                    )
                    return
                rows = []
                best_loss_so_far = float("inf")
                for step_i, extr_np in enumerate(best_hist):
                    # Overlay initial (blue edges) + best (red fill + edges)
                    mask_init = _render_mask_from_extrinsic(
                        self.optim_initial_extrinsic_guess, i_img
                    )
                    mask_best = _render_mask_from_extrinsic(extr_np, i_img)
                    base = self.images[i_img].copy()
                    # initial as blue edges
                    edges = cv2.Canny((mask_init.astype(np.uint8) * 255), 50, 150)
                    base[edges > 0] = (30, 144, 255)
                    # best as red fill
                    overlay_img = _overlay(
                        base, mask_best, color=(255, 0, 0), alpha=0.5
                    )
                    loss_i = float(losses[step_i]) if step_i < len(losses) else None
                    if loss_i is not None:
                        best_loss_so_far = min(best_loss_so_far, loss_i)
                    tx, ty, tz = extr_np[:3, 3].tolist()
                    rows.append([step_i, loss_i, best_loss_so_far, tx, ty, tz])
                    yield (
                        overlay_img,
                        f"step {step_i + 1}/{best_hist.shape[0]}",
                        step_i,
                        init_text,
                        rows,
                    )

            # Wiring
            idx.change(load_image, inputs=idx, outputs=image_view)
            image_view.select(on_select, inputs=[idx, label_radio], outputs=status)
            btn_clear.click(clear_points, inputs=idx, outputs=[image_view, status])
            btn_prev.click(prev, inputs=idx, outputs=[idx, image_view, status])
            btn_next.click(next_, inputs=idx, outputs=[idx, image_view, status])
            btn_generate.click(generate_mask, inputs=idx, outputs=[image_view, status])
            btn_accept.click(accept_mask, inputs=idx, outputs=[image_view, status])
            btn_save.click(save_masks, outputs=status)
            btn_exit.click(finish_and_exit, outputs=status)
            btn_preview.click(
                start_preview,
                inputs=opt_idx,
                outputs=[opt_image, opt_info, opt_idx, init_xyz, step_table],
            )

            # Initial image
            image_view.value = self.images[0]
            # Initial preview image
            opt_image.value = self.images[0]

        return demo

    def launch_and_wait(self) -> np.ndarray:
        self.app.launch(
            server_name=self.server_name,
            server_port=self.server_port,
            prevent_thread_lock=True,
        )
        # Wait for user to click Finish & Save
        while not self.done_event.is_set():
            time.sleep(0.2)
        try:
            self.app.close()
        except Exception:
            pass
        # Load saved masks if path provided; else build from self.masks
        if self.mask_path is not None and self.mask_path.exists():
            return np.load(self.mask_path)
        H, W = self.images[0].shape[:2]
        final_masks = []
        for i in range(self.num_images):
            if self.masks[i] is None:
                final_masks.append(np.zeros((H, W), dtype=np.float32))
            else:
                final_masks.append(self.masks[i].astype(np.float32))
        return np.stack(final_masks)


@dataclass
class SO101WebArgs(Args):
    """Web UI variant: Calibrate a (realsense) camera with LeRobot SO101.

    Uses a Gradio web app for interactive mask creation over SSH/headless.
    """

    output_dir: str = "results/so101"
    use_previous_captures: bool = False
    env_kwargs_json_path: Optional[str] = None

    # Web UI options
    server_port: int = 7860
    server_name: str = "0.0.0.0"
    opencv_intrinsics_path: Optional[str] = None
    opencv_hfov_deg: Optional[float] = None
    opencv_vfov_deg: Optional[float] = None


def main(args: SO101WebArgs):
    CALIBRATION_OFFSET = {
        "shoulder_pan": 0,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": 0,
        "gripper": 0,
    }
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            data = json.load(f)
            CALIBRATION_OFFSET = data.pop("calibration_offset")

    user_tuned_calibration_offset = any(
        CALIBRATION_OFFSET[k] != 0 for k in CALIBRATION_OFFSET.keys()
    )
    if not user_tuned_calibration_offset:
        logging.warning(
            "The calibration offset for sim2real/real2sim is not tuned!! Unless you are absolutely sure you will most likely get poor results."
        )

    robot: SO101Follower = create_real_robot("so101")
    robot_id = robot.id if robot.id is not None else "default"
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    robot.connect()

    cameras_ft = robot.cameras
    print(f"Found {len(cameras_ft)} cameras to calibrate")
    for k in cameras_ft.keys():
        (Path(args.output_dir) / robot_id / k).mkdir(parents=True, exist_ok=True)

    # Initial extrinsic guesses
    initial_extrinsic_guesses = dict()
    for k in cameras_ft.keys():
        initial_extrinsic_guess = np.eye(4)
        initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
        initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.5])
        initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)
        initial_extrinsic_guesses[k] = initial_extrinsic_guess

    print("Initial extrinsic guesses")
    for k in initial_extrinsic_guesses.keys():
        print(f"Camera {k}:\n{repr(initial_extrinsic_guesses[k])}")

    # Camera intrinsics
    intrinsics = dict()
    for cam_name, cam in robot.cameras.items():
        if isinstance(cam, RealSenseCamera):
            streams = cam.rs_profile.get_streams()
            assert len(streams) == 1, (
                "Only one stream per camera is supported at the moment and it must be the color steam. Make sure to not enable any other streams."
            )
            color_stream = streams[0]
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            intrinsics[cam_name] = np.array(
                [
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1],
                ]
            )
        else:
            try:
                # Hard-coded intrinsics for 5mm on IMX323 at 640x480
                # Base K_640x480 ≈ [[595.9, 0, 320], [0, 794.7, 240], [0, 0, 1]]
                frame = cam.async_read()
                height, width = frame.shape[:2]
                K_base = np.array(
                    [[595.9, 0.0, 320.0], [0.0, 794.7, 240.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                if width == 640 and height == 480:
                    intrinsics[cam_name] = K_base
                else:
                    # Scale hard-coded intrinsics if the resolution differs
                    intrinsics[cam_name] = scale_intrinsics(
                        K_base, 640, 480, width, height
                    )
                print(
                    f"[{cam_name}] Using hard-coded OpenCV intrinsics (5mm) scaled to {width}x{height}:\n{intrinsics[cam_name]}"
                )
            except Exception as e:
                print(f"Failed to set hard-coded OpenCV intrinsics for {cam_name}: {e}")
                pass

    # Collect data (images + link poses)
    joint_position_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    def get_qpos(robot: SO101Follower, flat: bool = True):
        obs = robot.bus.sync_read("Present_Position")
        for k in CALIBRATION_OFFSET.keys():
            obs[k] = obs[k] - CALIBRATION_OFFSET[k]
        for k in obs.keys():
            obs[k] = np.deg2rad(obs[k])
        if not flat:
            return obs
        joint_positions = []
        for k, v in obs.items():
            joint_positions.append(v)
        joint_positions = np.array(joint_positions)
        return joint_positions

    def set_target_qpos(robot: SO101Follower, qpos: np.ndarray):
        action = {}
        for name, qpos_val in zip(joint_position_names, qpos):
            action[name] = (
                np.rad2deg(qpos_val) + CALIBRATION_OFFSET[name.removesuffix(".pos")]
            )
        robot.send_action(action)

    robot_def_path = (
        Path(__file__).parent.parent / "assets" / "robots" / "so101" / "so101.urdf"
    )
    print(f"Robot definition path: {robot_def_path}")
    robot_urdf = URDF.load(str(robot_def_path))

    meshes = []
    mesh_link_names = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            geom = getattr(visual, "geometry", None)
            mesh_attr = getattr(geom, "mesh", None)
            if mesh_attr is None:
                continue
            link_meshes += mesh_attr.meshes
        if not link_meshes:
            continue
        merged = merge_meshes(link_meshes)
        if merged is None:
            continue
        if hasattr(merged, "vertices") and hasattr(merged, "faces"):
            meshes.append(merged)
            mesh_link_names.append(link.name)

    if (
        args.use_previous_captures
        and (Path(args.output_dir) / robot_id / "link_poses_dataset.npy").exists()
    ):
        link_poses_dataset = np.load(
            Path(args.output_dir) / robot_id / "link_poses_dataset.npy"
        )
        image_dataset = np.load(
            Path(args.output_dir) / robot_id / "image_dataset.npy", allow_pickle=True
        ).reshape(-1)[0]
    else:
        qpos_samples = [
            np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0.2]),
            np.array([np.pi / 3, -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0]),
        ]
        control_freq = 15
        max_radians_per_step = 0.05

        link_poses_dataset = np.zeros((len(qpos_samples), len(meshes), 4, 4))
        image_dataset = defaultdict(list)

        for i in range(len(qpos_samples)):
            goal_qpos = qpos_samples[i]
            target_qpos = get_qpos(robot)
            for _ in range(int(20 * control_freq)):
                start_loop_t = time.perf_counter()
                delta_qpos = goal_qpos - target_qpos
                delta_step = delta_qpos.clip(
                    min=-max_radians_per_step, max=max_radians_per_step
                )
                if np.linalg.norm(delta_qpos) < 1e-4:
                    break
                target_qpos += delta_step
                dt_s = time.perf_counter() - start_loop_t
                set_target_qpos(robot, target_qpos)
                time.sleep(1 / control_freq - dt_s)
            time.sleep(1)
            qpos_dict = get_qpos(robot, flat=False)
            for cam_name, cam in robot.cameras.items():
                image_dataset[cam_name].append(cam.async_read())
            cfg = dict()
            for k in robot_urdf.joint_map.keys():
                if k in qpos_dict:
                    cfg[k] = qpos_dict[k]
                else:
                    cfg[k] = 0.0
            link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
            for link_idx, link_name in enumerate(mesh_link_names):
                link_poses_dataset[i, link_idx] = link_poses[link_name]
        for k in image_dataset.keys():
            image_dataset[k] = np.stack(image_dataset[k])

        np.save(
            Path(args.output_dir) / robot_id / "link_poses_dataset.npy",
            link_poses_dataset,
        )
        np.save(Path(args.output_dir) / robot_id / "image_dataset.npy", image_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for k in initial_extrinsic_guesses.keys():
        print(f"Calibrating camera {k}")
        if k not in intrinsics:
            print(f"Skipping camera {k}: intrinsics not available")
            continue
        if k not in image_dataset:
            print(f"Skipping camera {k}: images not available")
            continue
        initial_extrinsic_guess = initial_extrinsic_guesses[k]
        intrinsic = intrinsics[k]
        images = image_dataset[k]
        camera_mount_poses = None
        camera_width = images.shape[2]
        camera_height = images.shape[1]

        mask_path = Path(args.output_dir) / robot_id / k / f"mask.npy"
        if args.use_previous_captures and mask_path.exists():
            print(f"Using previous mask from {mask_path}")
            masks = np.load(mask_path)
        else:
            print("Launching Gradio web UI for interactive mask creation...")
            annotator = WebMaskAnnotator(
                images=images,
                model_cfg=args.model_cfg,
                checkpoint=args.checkpoint,
                server_name=args.server_name,
                server_port=args.server_port,
                mask_path=mask_path,
                intrinsic=intrinsic,
                link_poses_dataset=link_poses_dataset,
                meshes=meshes,
                initial_extrinsic_guess=initial_extrinsic_guess,
                camera_mount_poses=camera_mount_poses,
                iterations=args.train_steps,
                early_stopping_steps=args.early_stopping_steps,
            )
            masks = annotator.launch_and_wait()
            np.save(mask_path, masks)

        predicted_camera_extrinsic_opencv = (
            optimize(
                camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
                masks=torch.from_numpy(masks).float().to(device),
                link_poses_dataset=torch.from_numpy(link_poses_dataset)
                .float()
                .to(device),
                initial_extrinsic_guess=torch.tensor(initial_extrinsic_guess)
                .float()
                .to(device),
                meshes=meshes,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_mount_poses=(
                    torch.from_numpy(camera_mount_poses).float().to(device)
                    if camera_mount_poses is not None
                    else None
                ),
                gt_camera_pose=None,
                iterations=args.train_steps,
                early_stopping_steps=args.early_stopping_steps,
            )
            .cpu()
            .numpy()
        )
        predicted_camera_extrinsic_ros = opencv2ros(predicted_camera_extrinsic_opencv)

        print("Predicted camera extrinsic")
        print(f"OpenCV:\n{repr(predicted_camera_extrinsic_opencv)}")
        print(
            f"ROS/SAPIEN/ManiSkill/Mujoco/Isaac:\n{repr(predicted_camera_extrinsic_ros)}"
        )

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            Path(args.output_dir) / robot_id / k / "camera_extrinsic_opencv.npy",
            predicted_camera_extrinsic_opencv,
        )
        np.save(
            Path(args.output_dir) / robot_id / k / "camera_extrinsic_ros.npy",
            predicted_camera_extrinsic_ros,
        )
        np.save(
            Path(args.output_dir) / robot_id / k / "camera_intrinsic.npy", intrinsic
        )
        np.save(
            Path(args.output_dir) / robot_id / k / "camera_intrinsic_128x128.npy",
            scale_intrinsics(
                intrinsic, images[0].shape[1], images[0].shape[0], 128, 128
            ),
        )

        visualize_extrinsic_results_red_mask(
            images=images,
            link_poses_dataset=link_poses_dataset,
            meshes=meshes,
            intrinsic=intrinsic,
            extrinsics=np.stack(
                [initial_extrinsic_guess, predicted_camera_extrinsic_opencv]
            ),
            masks=masks,
            labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
            output_dir=str(Path(args.output_dir) / robot_id / k),
            mask_color=(255, 0, 0),
        )
        print(f"Visualizations saved to {Path(args.output_dir) / robot_id / k}")


if __name__ == "__main__":
    main(tyro.cli(SO101WebArgs))
