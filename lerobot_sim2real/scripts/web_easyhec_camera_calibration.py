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
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from transforms3d.euler import euler2mat
from easyhec.examples.real.base import Args
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv

# Import shared URDF utilities
from lerobot_sim2real.utils.urdf_utils import load_robot_meshes_for_calibration

# Optional: red mask overlay visualization
from lerobot_sim2real.utils.custom_visualization import (
    visualize_extrinsic_results_red_mask,
)

from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.utils.camera import scale_intrinsics
from lerobot_sim2real.optim.optimize_with_better_logging import optimize


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
        # Optional optimization context for running optimization
        intrinsic: Optional[np.ndarray] = None,
        link_poses_dataset: Optional[np.ndarray] = None,
        meshes: Optional[List] = None,
        initial_extrinsic_guess: Optional[np.ndarray] = None,
        camera_mount_poses: Optional[np.ndarray] = None,
        iterations: int = 10000,
        early_stopping_steps: int = 10000,
        output_dir: Optional[Path] = None,
        initial_extrinsic_x: float = 0.0,
        initial_extrinsic_y: float = 0.0,
        initial_extrinsic_z: float = 0.0,
        initial_extrinsic_x_rotation: float = 0.0,
        initial_extrinsic_y_rotation: float = 0.0,
        initial_extrinsic_z_rotation: float = 0.0,
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
        self.output_dir = output_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_extrinsic_x = initial_extrinsic_x
        self.initial_extrinsic_y = initial_extrinsic_y
        self.initial_extrinsic_z = initial_extrinsic_z
        self.initial_extrinsic_x_rotation = initial_extrinsic_x_rotation
        self.initial_extrinsic_y_rotation = initial_extrinsic_y_rotation
        self.initial_extrinsic_z_rotation = initial_extrinsic_z_rotation

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
            gr.Markdown("# Simple EasyHEC")
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

                    # initial_extrinsic_guess[:3, :3] = euler2mat(0.26, 0.11, 0.86)
                    # initial_extrinsic_guess[:3, 3] = np.array([0.0, 0.67, -1.66])

                    # Initial extrinsic guess position controls
                    gr.Markdown("## Initial Extrinsic Guess Position (ROS coordinates)")
                    with gr.Row():
                        # More realistic defaults for overhead camera
                        pos_x = gr.Number(
                            value=self.initial_extrinsic_x,
                            label="X Position (forward)",
                            step=0.01,
                        )
                        pos_y = gr.Number(
                            value=self.initial_extrinsic_y,
                            label="Y Position (left)",
                            step=0.01,
                        )
                        pos_z = gr.Number(
                            value=self.initial_extrinsic_z,
                            label="Z Position (up)",
                            step=0.01,
                        )

                    # Initial extrinsic guess rotation controls
                    gr.Markdown("## Initial Extrinsic Guess Rotation (radians)")
                    with gr.Row():
                        rot_x = gr.Number(
                            value=self.initial_extrinsic_x_rotation,
                            label="Roll (X)",
                            step=0.01,
                        )  # -30 degrees
                        rot_y = gr.Number(
                            value=self.initial_extrinsic_y_rotation,
                            label="Pitch (Y)",
                            step=0.01,
                        )
                        rot_z = gr.Number(
                            value=self.initial_extrinsic_z_rotation,
                            label="Yaw (Z)",
                            step=0.01,
                        )

                    btn_prev = gr.Button("Prev")
                    btn_next = gr.Button("Next")
                    btn_render_initial_extrinsic_guess = gr.Button(
                        "Render Initial Extrinsic Guess"
                    )
                    btn_clear = gr.Button("Clear Points")
                    btn_generate = gr.Button("Generate Mask")
                    btn_accept = gr.Button("Accept Mask")
                    btn_save = gr.Button("Save Masks")
                    btn_exit = gr.Button("Finish & Exit")

            with gr.Row():
                btn_run = gr.Button("Run Optimization")

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

            def render_initial_extrinsic_guess_before_optimization(
                i: float,
                x_pos: float,
                y_pos: float,
                z_pos: float,
                rot_x_rad: float,
                rot_y_rad: float,
                rot_z_rad: float,
            ) -> np.ndarray:
                """Render the initial extrinsic guess on the current image."""
                i = int(i)

                if (
                    self.optim_intrinsic is None
                    or self.optim_link_poses_dataset is None
                    or self.optim_initial_extrinsic_guess is None
                    or self.optim_meshes is None
                ):
                    # Return original image if optimization context not available
                    return self.images[i]

                # Get current image
                current_image = self.images[i : i + 1]  # Single image as batch
                current_link_poses = self.optim_link_poses_dataset[
                    i : i + 1
                ]  # Single pose set

                # Create a custom initial extrinsic with user-specified position and rotation
                custom_extrinsic = np.eye(4)

                # Convert degrees to radians and create rotation matrix
                # rot_x_rad = np.deg2rad(rot_x_deg)

                # rot_y_rad = np.deg2rad(rot_y_deg)
                # rot_z_rad = np.deg2rad(rot_z_deg)
                custom_extrinsic[:3, :3] = euler2mat(rot_x_rad, rot_y_rad, rot_z_rad)
                # custom_extrinsic[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
                custom_extrinsic[:3, 3] = np.array([x_pos, y_pos, z_pos])

                # Convert from ROS to OpenCV coordinate system
                custom_extrinsic = ros2opencv(custom_extrinsic)

                # Create a single extrinsic for visualization
                extrinsics_vis = custom_extrinsic[None, ...]  # Add batch dimension

                # Use the visualization function to render the meshes
                try:
                    # Create a temporary output directory for the visualization
                    import tempfile

                    with tempfile.TemporaryDirectory() as temp_dir:
                        rendered_image = visualize_extrinsic_results_red_mask(
                            images=current_image,
                            link_poses_dataset=current_link_poses,
                            meshes=self.optim_meshes,
                            intrinsic=self.optim_intrinsic,
                            extrinsics=extrinsics_vis,
                            masks=None,  # No masks for initial guess visualization
                            labels=["Initial Extrinsic Guess"],
                            output_dir=temp_dir,  # Use temp directory
                            return_rgb=True,  # Return RGB array
                            mask_color=None,  # No mask overlay
                            mask_colors=[(255, 0, 0)],  # Red color for initial guess
                            invert_extrinsic=True,
                        )

                    # Return the rendered image
                    if rendered_image is not None:
                        return rendered_image
                    else:
                        return self.images[i]

                except Exception as e:
                    print(f"Error rendering initial extrinsic guess: {e}")
                    return self.images[i]

            def run_optimization() -> str:
                if (
                    self.optim_intrinsic is None
                    or self.optim_link_poses_dataset is None
                    or self.optim_initial_extrinsic_guess is None
                    or self.optim_meshes is None
                ):
                    return "Optimization context not available yet."

                # Build masks tensor from current UI state; fill missing with zeros
                H, W = self.images[0].shape[:2]
                mask_list = []
                for i_img in range(self.num_images):
                    if self.masks[i_img] is None:
                        mask_list.append(np.zeros((H, W), dtype=np.float32))
                    else:
                        mask_list.append(self.masks[i_img].astype(np.float32))
                masks_np = np.stack(mask_list)
                if self.mask_path is not None:
                    self.mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.mask_path, masks_np)

                # Prepare output directory
                out_dir = (
                    self.output_dir
                    if self.output_dir is not None
                    else (
                        self.mask_path.parent
                        if self.mask_path is not None
                        else Path(".")
                    )
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                history = optimize(
                    camera_intrinsic=torch.from_numpy(self.optim_intrinsic)
                    .float()
                    .to(device),
                    masks=torch.from_numpy(masks_np).float().to(device),
                    link_poses_dataset=torch.from_numpy(self.optim_link_poses_dataset)
                    .float()
                    .to(device),
                    initial_extrinsic_guess=torch.from_numpy(
                        self.optim_initial_extrinsic_guess
                    )
                    .float()
                    .to(device),
                    meshes=self.optim_meshes,
                    camera_width=W,
                    camera_height=H,
                    camera_mount_poses=None,
                    gt_camera_pose=None,
                    iterations=self.optim_iterations,
                    early_stopping_steps=self.optim_early_stopping,
                    return_history=True,
                    learning_rate=0.0001,
                    batch_size=None,
                )

                # Use the last best extrinsic
                predicted_camera_extrinsic_opencv = (
                    history["best_extrinsics"][-1].cpu().numpy()
                )
                np.save(
                    out_dir / "camera_extrinsic_opencv.npy",
                    predicted_camera_extrinsic_opencv,
                )
                predicted_camera_extrinsic_ros = opencv2ros(
                    predicted_camera_extrinsic_opencv
                )
                np.save(
                    out_dir / "camera_extrinsic_ros.npy", predicted_camera_extrinsic_ros
                )
                np.save(out_dir / "camera_intrinsic.npy", self.optim_intrinsic)

                # Build history-based visualization
                best_extrinsics_np = history["best_extrinsics"].cpu().numpy()
                steps = np.array(history["best_extrinsics_step"], dtype=int)
                losses = np.array(history["best_extrinsics_losses"], dtype=float)
                m = len(steps)
                if m <= 50:
                    sel = np.arange(m)
                else:
                    sel = np.unique(np.round(np.linspace(0, m - 1, 50)).astype(int))
                extrinsics_vis = best_extrinsics_np[sel]
                labels = [
                    f"Step {int(steps[k])} (loss={float(losses[k]):.6f})" for k in sel
                ]
                extrinsics_vis = np.concatenate(
                    [self.optim_initial_extrinsic_guess[None, ...], extrinsics_vis],
                    axis=0,
                )
                labels = ["Initial Guess"] + labels
                per_extrinsic_colors = [(30, 144, 255)] + [
                    (0, 255, 0) for _ in range(len(extrinsics_vis) - 1)
                ]

                _ = visualize_extrinsic_results_red_mask(
                    images=self.images,
                    link_poses_dataset=self.optim_link_poses_dataset,
                    meshes=self.optim_meshes,
                    intrinsic=self.optim_intrinsic,
                    extrinsics=extrinsics_vis,
                    masks=masks_np,
                    labels=labels,
                    output_dir=str(out_dir),
                    return_rgb=False,
                    mask_color=(255, 0, 0),
                    mask_colors=per_extrinsic_colors,
                    invert_extrinsic=True,
                    create_compact_version=True,
                )

                return f"Optimization complete. Saved results to {out_dir}"

            # Wiring
            idx.change(load_image, inputs=idx, outputs=image_view)
            image_view.select(on_select, inputs=[idx, label_radio], outputs=status)
            btn_render_initial_extrinsic_guess.click(
                render_initial_extrinsic_guess_before_optimization,
                inputs=[idx, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z],
                outputs=image_view,
            )
            btn_clear.click(clear_points, inputs=idx, outputs=[image_view, status])
            btn_prev.click(prev, inputs=idx, outputs=[idx, image_view, status])
            btn_next.click(next_, inputs=idx, outputs=[idx, image_view, status])
            btn_generate.click(generate_mask, inputs=idx, outputs=[image_view, status])
            btn_accept.click(accept_mask, inputs=idx, outputs=[image_view, status])
            btn_save.click(save_masks, outputs=status)
            btn_exit.click(finish_and_exit, outputs=status)
            btn_run.click(run_optimization, outputs=status)

            # Initial image
            image_view.value = self.images[0]
            # No preview image section

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

    # Training configuration - override base class defaults
    train_steps: int = 2**14
    """number of optimization steps"""
    early_stopping_steps: int = 2**14
    """if after this many steps of optimization the loss has not improved, then optimization will stop"""

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

    uid = "so101"
    urdf_name = "so101"

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            data = json.load(f)
            CALIBRATION_OFFSET = data.pop("calibration_offset")

    user_tuned_calibration_offset = any(
        CALIBRATION_OFFSET[k] != 0 for k in CALIBRATION_OFFSET.keys()
    )

    print(f"user_tuned_calibration_offset: {user_tuned_calibration_offset}")
    print(f"CALIBRATION_OFFSET: {CALIBRATION_OFFSET}")
    if not user_tuned_calibration_offset:
        logging.warning(
            "The calibration offset for sim2real/real2sim is not tuned!! Unless you are absolutely sure you will most likely get poor results."
        )

    robot: SO101Follower = create_real_robot(uid)
    robot_id = robot.id if robot.id is not None else "default"
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    robot.connect()

    cameras_ft = robot.cameras
    print(f"Found {len(cameras_ft)} cameras to calibrate")
    for k in cameras_ft.keys():
        (Path(args.output_dir) / robot_id / k).mkdir(parents=True, exist_ok=True)

    initial_extrinsic_x = 0.265
    initial_extrinsic_y = 0.15
    initial_extrinsic_z = 0.79

    initial_extrinsic_x_rotation = 0.22
    initial_extrinsic_y_rotation = 0.0
    initial_extrinsic_z_rotation = -1.57

    # Initial extrinsic guesses
    initial_extrinsic_guesses = dict()
    for k in cameras_ft.keys():
        initial_extrinsic_guess = np.eye(4)

        # This should be a rough guess of your camera position and orientation relative to the robot base
        # Position camera above and slightly forward of robot, angled downward
        # ROS coordinates: X=forward, Y=left, Z=up
        initial_extrinsic_guess[:3, :3] = euler2mat(
            initial_extrinsic_x_rotation,
            initial_extrinsic_y_rotation,
            initial_extrinsic_z_rotation,
        )  # Rotation matrix from Euler angles
        initial_extrinsic_guess[:3, 3] = np.array(
            [
                initial_extrinsic_x,
                initial_extrinsic_y,
                initial_extrinsic_z,
            ]
        )  # Position coordinates

        # Directly on the base should wash out the camera with the mesh
        # initial_extrinsic_guess[:3, 3] = np.array([0.0, 0.0, 0.0])
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

    def get_qpos(robot: SO101Follower | SO100Follower, flat: bool = True):
        obs = robot.bus.sync_read("Present_Position")
        for k in CALIBRATION_OFFSET.keys():
            obs[k] = obs[k] - CALIBRATION_OFFSET[k]
        for k in obs.keys():
            obs[k] = np.deg2rad(obs[k])
        if not flat:
            return obs
        # joint_positions = []
        # for k, v in obs.items():
        #     joint_positions.append(v)
        # joint_positions = np.array(joint_positions)
        # return joint_positions

        # Return joints in the explicit control order used by set_target_qpos
        ordered_positions = []
        for name in joint_position_names:
            base = name.removesuffix(".pos")
            if base in obs:
                ordered_positions.append(obs[base])
            else:
                ordered_positions.append(0.0)
        return np.array(ordered_positions)

    def set_target_qpos(robot: SO101Follower | SO100Follower, qpos: np.ndarray):
        action = {}
        for name, qpos_val in zip(joint_position_names, qpos):
            print(
                f"Setting {name} to {qpos_val} + {CALIBRATION_OFFSET[name.removesuffix('.pos')]}"
            )
            action[name] = (
                np.rad2deg(qpos_val) + CALIBRATION_OFFSET[name.removesuffix(".pos")]
            )
        robot.send_action(action)

    # robot_def_path = (
    #     Path(__file__).parent.parent / "assets" / "robots" / "so101" / "so101.urdf"
    # )
    # print(f"Robot definition path: {robot_def_path}")
    # robot_urdf = URDF.load(str(robot_def_path))

    # Use shared utility function to load URDF and extract meshes for calibration
    robot_urdf, meshes, mesh_link_names = load_robot_meshes_for_calibration(urdf_name)

    # print(f"Meshes: {meshes}")
    print(f"Mesh link names: {mesh_link_names}")

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
            np.array([0, 0, 0, np.pi / 2, 0, 0]),
            np.array([np.pi / 4, -np.pi / 6, 0, np.pi / 2, np.pi / 6, np.pi / 3]),
            np.array([-(np.pi / 4), -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0]),
            # np.array([-np.pi / 4, -np.pi / 6, np.pi / 6, np.pi / 2, np.pi / 2, 0.1]),
            # np.array([0, 0, 0, 0, np.pi / 2, 0.2]),
            # np.array([0, np.pi / 6, 0, 0, np.pi / 2, 0.2]),
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

        mask_path = Path(args.output_dir) / robot_id / k / "mask.npy"
        if args.use_previous_captures and mask_path.exists():
            print(f"Using previous mask from {mask_path}")
            masks = np.load(mask_path)
        else:
            print("Launching Gradio web UI for interactive mask creation...")

            # Extract position and rotation from the initial guess for UI defaults
            # Convert back from OpenCV to ROS to get the original values
            from easyhec.utils.camera_conversions import opencv2ros
            from scipy.spatial.transform import Rotation as R

            initial_guess_ros = opencv2ros(initial_extrinsic_guess)
            initial_pos = initial_guess_ros[:3, 3]
            print(f"Initial position: {initial_pos}")
            initial_rot_matrix = initial_guess_ros[:3, :3]
            initial_rot_euler = R.from_matrix(initial_rot_matrix).as_euler("xyz")
            print(f"Initial rotation: {initial_rot_euler}")

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
                output_dir=Path(args.output_dir) / robot_id / k,
                initial_extrinsic_x=float(initial_pos[0]),
                initial_extrinsic_y=float(initial_pos[1]),
                initial_extrinsic_z=float(initial_pos[2]),
                initial_extrinsic_x_rotation=float(initial_rot_euler[0]),
                initial_extrinsic_y_rotation=float(initial_rot_euler[1]),
                initial_extrinsic_z_rotation=float(initial_rot_euler[2]),
            )
            masks = annotator.launch_and_wait()
            np.save(mask_path, masks)

        # Run optimization with history to build visualization
        history = optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            masks=torch.from_numpy(masks).float().to(device),
            link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
            initial_extrinsic_guess=torch.from_numpy(initial_extrinsic_guess)
            .float()
            .to(device),
            meshes=meshes,
            camera_width=camera_width,
            camera_height=camera_height,
            iterations=args.train_steps,
            early_stopping_steps=args.early_stopping_steps,
            return_history=True,
        )

        predicted_camera_extrinsic_opencv = history["best_extrinsics"][-1].cpu().numpy()
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

        # Build history-based visualization similar to opencv_paper_calibration
        best_extrinsics_np = history["best_extrinsics"].cpu().numpy()
        steps = np.array(history["best_extrinsics_step"], dtype=int)
        losses = np.array(history["best_extrinsics_losses"], dtype=float)
        m = len(steps)
        if m <= 50:
            sel = np.arange(m)
        else:
            sel = np.unique(np.round(np.linspace(0, m - 1, 50)).astype(int))
        extrinsics_vis = best_extrinsics_np[sel]
        labels = [f"Step {int(steps[t])} (loss={float(losses[t]):.6f})" for t in sel]
        extrinsics_vis = np.concatenate(
            [initial_extrinsic_guess[None, ...], extrinsics_vis], axis=0
        )
        labels = ["Initial Extrinsic Guess"] + labels
        per_extrinsic_colors = [(30, 144, 255)] + [
            (0, 255, 0) for _ in range(len(extrinsics_vis) - 1)
        ]

        visualize_extrinsic_results_red_mask(
            images=images,
            link_poses_dataset=link_poses_dataset,
            meshes=meshes,
            intrinsic=intrinsic,
            extrinsics=extrinsics_vis,
            masks=masks,
            labels=labels,
            output_dir=str(Path(args.output_dir) / robot_id / k),
            return_rgb=False,
            mask_color=(255, 0, 0),
            mask_colors=per_extrinsic_colors,
            invert_extrinsic=True,
            create_compact_version=True,
        )
        print(f"Visualizations saved to {Path(args.output_dir) / robot_id / k}")


if __name__ == "__main__":
    main(tyro.cli(SO101WebArgs))
