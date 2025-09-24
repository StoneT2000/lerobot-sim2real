"""OpenCV paper-based camera calibration using EasyHEC.

This script captures a single RGB frame from an OpenCV camera, lets you segment
the paper (letter or A4) interactively, and optimizes the camera extrinsic to
align the rendered paper mesh with your mask.

Based on conventions used in web_easyhec_camera_calibration.py:
- Build initial extrinsic in ROS convention, convert to OpenCV via ros2opencv
- Optimize in OpenCV convention; convert result back to ROS for saving
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import time
import numpy as np
import torch
import trimesh
import tyro
from transforms3d.euler import euler2mat

from easyhec.examples.real.base import Args

# from lerobot.cameras.realsense import RealSenseCamera
from lerobot.cameras.realsense import RealSenseCameraConfig, RealSenseCamera

# from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from lerobot_sim2real.utils.camera import scale_intrinsics
import gradio as gr
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# For multi-start optimization with loss tracking
from lerobot_sim2real.optim.optimize_with_better_logging import optimize
# Defer importing custom_visualization until runtime to avoid sys.path issues


@dataclass
class OpenCVPaperArgs(Args):
    """Calibrate an OpenCV camera with a piece of standard-sized paper.

    - Provide intrinsics via npy file or approximate via FOVs.
    - Captures one RGB frame from a cv2.VideoCapture device.
    """

    output_dir: str = "results/statement"
    paper_type: str = "statement"  # or "letter" or "a4"

    # OpenCV camera settings
    device_index: int = 0
    camera_width = 480
    camera_height = 640

    # Intrinsics sources (choose one)
    opencv_intrinsics_path: Optional[str] = None  # path to .npy K (3x3)
    hfov_deg: Optional[float] = None
    vfov_deg: Optional[float] = None

    # Web UI options
    server_port: int = 7860
    server_name: str = "0.0.0.0"


PAPER_SIZES_M = {
    # Half-letter (aka Statement): 5.5 x 8.5 inches
    "statement": {"width": 0.1397, "height": 0.2159},
    # US Letter: 8.5 x 11 inches
    "letter": {"width": 0.2159, "height": 0.2794},
    # ISO A4: 210 x 297 mm
    "a4": {"width": 0.2100, "height": 0.2970},
}


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


def _k_from_fov(
    width: int, height: int, hfov_deg: Optional[float], vfov_deg: Optional[float]
) -> Optional[np.ndarray]:
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
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_or_build_intrinsics_for_size() -> np.ndarray:
    # K = _k_from_fov(width, height, args.hfov_deg, args.vfov_deg)
    # if K is not None:
    #     return K
    # Fallback: approximate from a base K (640x480) scaled to actual resolution
    # K_base = np.array(
    #     [[595.9, 0.0, 320.0], [0.0, 794.7, 240.0], [0.0, 0.0, 1.0]],
    #     dtype=np.float32,
    # )
    # return scale_intrinsics(K_base, 640, 480, width, height)
    return get_intrinics("realsense")


class PaperWebAnnotator:
    """Minimal Gradio UI for SAM2 point-and-click mask creation on a single image."""

    def __init__(
        self,
        image: np.ndarray,
        model_cfg: str,
        checkpoint: str,
        # Optimization context
        intrinsic_K: np.ndarray,
        initial_extrinsic_opencv: np.ndarray,
        link_poses_dataset: np.ndarray,
        meshes: List[trimesh.Trimesh],
        camera_width: int,
        camera_height: int,
        train_steps: int,
        early_stopping_steps: int,
        output_dir: Path,
        paper_type: str,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
    ) -> None:
        self.image = image
        self.server_name = server_name
        self.server_port = server_port
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=device_str)
        )
        self.clicked_points: List[Tuple[int, int, int]] = []  # (x, y, label)
        self.mask: Optional[np.ndarray] = None
        # Optimization context
        self.K = intrinsic_K.astype(np.float32)
        self.initial_extrinsic = initial_extrinsic_opencv.astype(np.float32)
        self.link_poses_dataset = link_poses_dataset.astype(np.float32)
        self.meshes = meshes
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.train_steps = int(train_steps)
        self.early_stopping_steps = int(early_stopping_steps)
        self.output_dir = output_dir
        self.paper_type = paper_type
        self.optimization_done: bool = False
        self.finished: bool = False
        self._result_images: List[np.ndarray] = []
        self._build_ui()

    @staticmethod
    def _overlay_mask(
        image: np.ndarray, mask: Optional[np.ndarray], color=(255, 0, 0), alpha=0.6
    ) -> np.ndarray:
        if mask is None:
            return image
        mask_b = mask.astype(bool)
        result = image.copy()
        overlay = np.full_like(result, color)
        result[mask_b] = (
            (1 - alpha) * result[mask_b] + alpha * overlay[mask_b]
        ).astype(np.uint8)
        return result

    @staticmethod
    def _composite_rgba_over_rgb(
        foreground_rgba: np.ndarray, background_rgb: np.ndarray
    ) -> np.ndarray:
        # Ensure shapes
        fh, fw = foreground_rgba.shape[:2]
        bh, bw = background_rgb.shape[:2]
        if (fh != bh) or (fw != bw):
            foreground_rgba = cv2.resize(
                foreground_rgba, (bw, bh), interpolation=cv2.INTER_LINEAR
            )
        if foreground_rgba.shape[2] == 3:
            return foreground_rgba
        fg_rgb = foreground_rgba[:, :, :3].astype(np.float32)
        alpha = foreground_rgba[:, :, 3:4].astype(np.float32) / 255.0
        bg_rgb = background_rgb.astype(np.float32)
        out = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _build_ui(self) -> None:
        with gr.Blocks() as demo:
            gr.Markdown("# Paper Mask Annotation (SAM2)")
            with gr.Row():
                with gr.Column(scale=3):
                    image_view = gr.Image(
                        value=self.image, label="Image", interactive=True, type="numpy"
                    )
                with gr.Column():
                    label_radio = gr.Radio(
                        [1, -1], value=1, label="Point Label (1=pos, -1=neg)"
                    )
                    status = gr.Textbox(label="Status", interactive=False)
                    interval_input = gr.Number(
                        value=100, label="Visualization interval (steps)", precision=0
                    )
                    btn_clear = gr.Button("Clear Points")
                    btn_generate = gr.Button("Generate Mask")
                    btn_run = gr.Button("Run Optimization")
                    btn_finish = gr.Button("Finish & Exit")

            # No preview; results saved to disk only

            def on_select(evt: gr.SelectData, lbl: int) -> str:
                x, y = evt.index[0], evt.index[1]
                self.clicked_points.append((int(x), int(y), int(lbl)))
                return f"Points: {len(self.clicked_points)}"

            def clear_points() -> Tuple[np.ndarray, str]:
                self.clicked_points = []
                self.mask = None
                return self.image, "Cleared points"

            def generate_mask(lbl: int) -> Tuple[np.ndarray, str]:
                if len(self.clicked_points) == 0:
                    return self.image, "Add at least one point before generating mask"
                input_label = np.array(
                    [p[2] for p in self.clicked_points], dtype=np.int32
                )
                input_point = np.array(
                    [[p[0], p[1]] for p in self.clicked_points], dtype=np.float32
                )
                with torch.inference_mode():
                    self.predictor.set_image(self.image)
                    mask, _, _ = self.predictor.predict(
                        input_point, input_label, multimask_output=False
                    )
                self.mask = mask[0]
                return PaperWebAnnotator._overlay_mask(self.image, self.mask), (
                    f"Mask generated ({int(self.mask.sum())} px)"
                )

            def run_optimization(interval_val: float) -> Tuple[str]:
                if self.mask is None:
                    return ("Generate a mask first.",)
                # Lazy import custom visualizer; add repo root to sys.path if needed
                _vis_extrinsic_red_mask = None
                try:
                    from custom_visualization import (
                        visualize_extrinsic_results_red_mask as _vis_extrinsic_red_mask,
                    )
                except Exception:
                    import sys as _sys
                    from pathlib import Path as _Path

                    try:
                        _sys.path.append(str(_Path(__file__).resolve().parents[2]))
                        from custom_visualization import (
                            visualize_extrinsic_results_red_mask as _vis_extrinsic_red_mask,
                        )
                    except Exception:
                        _vis_extrinsic_red_mask = None
                # Prepare IO
                self.output_dir.mkdir(parents=True, exist_ok=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                history = optimize(
                    camera_intrinsic=torch.from_numpy(self.K).float().to(device),
                    masks=torch.from_numpy(np.stack([self.mask])).float().to(device),
                    link_poses_dataset=torch.from_numpy(self.link_poses_dataset)
                    .float()
                    .to(device),
                    initial_extrinsic_guess=torch.from_numpy(self.initial_extrinsic)
                    .float()
                    .to(device),
                    meshes=self.meshes,
                    camera_width=self.camera_width,
                    camera_height=self.camera_height,
                    camera_mount_poses=None,
                    gt_camera_pose=None,
                    iterations=self.train_steps,
                    early_stopping_steps=self.early_stopping_steps,
                    return_history=True,
                )

                print("optimization completed, saving results and visualization")

                # Take the last best item
                predicted_camera_extrinsic_opencv = (
                    history["best_extrinsics"][-1].cpu().numpy()
                )

                # Save npy results
                np.save(
                    self.output_dir / "camera_extrinsic_opencv.npy",
                    predicted_camera_extrinsic_opencv,
                )
                predicted_camera_extrinsic_ros = opencv2ros(
                    predicted_camera_extrinsic_opencv
                )
                np.save(
                    self.output_dir / "camera_extrinsic_ros.npy",
                    predicted_camera_extrinsic_ros,
                )
                np.save(self.output_dir / "camera_intrinsic.npy", self.K)

                print("_vis_extrinsic_red_mask:", _vis_extrinsic_red_mask)

                # Evenly sample up to 50 best extrinsics and label with step and loss
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
                    f"Step {int(steps[k])} (loss={float(losses[k]):.2f})" for k in sel
                ]

                # Prepend initial guess (blue) and set all steps to green
                extrinsics_vis = np.concatenate(
                    [self.initial_extrinsic[None, ...], extrinsics_vis], axis=0
                )
                labels = ["Initial Guess"] + labels
                per_extrinsic_colors = [(30, 144, 255)] + [
                    (0, 255, 0) for _ in range(len(extrinsics_vis) - 1)
                ]

                _ = _vis_extrinsic_red_mask(
                    images=np.stack([self.image]),
                    link_poses_dataset=self.link_poses_dataset,
                    meshes=self.meshes,
                    intrinsic=self.K,
                    extrinsics=extrinsics_vis,
                    masks=np.stack([self.mask]),
                    labels=labels,
                    output_dir=str(self.output_dir),
                    return_rgb=False,
                    mask_color=(255, 0, 0),
                    mask_colors=per_extrinsic_colors,
                )

                # Save visualization, then copy to timestamped paper_type filename
                ts = time.strftime("%Y%m%d_%H%M%S")
                saved_name = f"{self.paper_type}_{ts}.png"
                viz_path = self.output_dir / "0.png"
                if viz_path.exists():
                    png = cv2.imread(str(viz_path), cv2.IMREAD_COLOR)
                    if png is not None:
                        cv2.imwrite(str(self.output_dir / saved_name), png)

                self.optimization_done = True
                status_msg = (
                    f"Optimization complete. Saved: {self.output_dir / saved_name}"
                )
                return (status_msg,)

            def finish_and_exit() -> str:
                self.finished = True
                return "Exiting. Calibration session finished."

            image_view.select(on_select, inputs=[label_radio], outputs=status)
            btn_clear.click(clear_points, outputs=[image_view, status])
            btn_generate.click(
                generate_mask, inputs=[label_radio], outputs=[image_view, status]
            )
            btn_run.click(run_optimization, inputs=[interval_input], outputs=[status])
            btn_finish.click(finish_and_exit, outputs=status)

        self.app = demo

    def launch_and_wait(self) -> None:
        self.app.launch(
            server_name=self.server_name,
            server_port=self.server_port,
            prevent_thread_lock=True,
        )
        try:
            while not self.finished:
                time.sleep(0.2)
        finally:
            try:
                self.app.close()
            except Exception:
                pass


def _capture_one_frame(args: OpenCVPaperArgs) -> np.ndarray:
    cap = cv2.VideoCapture(args.device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.device_index}")
    # Request resolution (may not be honored by driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    # Warm up
    for _ in range(30):
        _ = cap.read()
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError("Failed to capture frame from OpenCV camera")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


def _capture_one_frame_realsense(args: OpenCVPaperArgs) -> np.ndarray:
    """
    Camera #2:
    Name: Intel RealSense D435
    Type: RealSense
    Id: 831612073213
    Firmware version: 5.10.13
    Usb type descriptor: 3.2
    Physical port: /sys/devices/pci0000:00/0000:00:14.0/usb2/2-2/2-2:1.0/video4linux/video4
    Product id: 0B07
    Product line: D400
    Default stream profile:
      Stream_type: Color
      Format: rgb8
      Width: 640
      Height: 480
      Fps: 30
    """
    camera = RealSenseCamera(
        RealSenseCameraConfig("831612073213", fps=30, width=640, height=480)
    )
    camera.connect()
    frame = camera.async_read()
    camera.disconnect()
    return frame


def get_intrinics(camera_type: str) -> np.ndarray:
    if camera_type == "realsense":
        camera = RealSenseCamera(
            RealSenseCameraConfig("831612073213", fps=30, width=640, height=480)
        )
        camera.connect()
        streams = camera.rs_profile.get_streams()
        assert len(streams) == 1, (
            "Only one stream per camera is supported at the moment and it must be the color steam. Make sure to not enable any other streams."
        )
        color_stream = streams[0]
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        intrinsics = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ]
        )
        camera.disconnect()
        return intrinsics

    else:
        initial_extrinsic_guess = np.eye(4, dtype=np.float32)
        initial_extrinsic_guess[:3, :3] = euler2mat(0.0, np.pi / 4.0, 0.0)
        initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.4], dtype=np.float32)
        initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)
        return initial_extrinsic_guess


def main(args: OpenCVPaperArgs) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Capture one RGB image first to know actual resolution
    print("Capturing one frame from camera...")
    image = _capture_one_frame_realsense(args)
    camera_height, camera_width = image.shape[0], image.shape[1]
    # Build intrinsics for the actual resolution (file → FOV → scaled base K)
    K = _load_or_build_intrinsics_for_size()

    print(f"Camera intrinsics (OpenCV):\n{repr(K)}")

    # Initial extrinsic guess (ROS convention), then convert to OpenCV
    initial_extrinsic_guess = np.eye(4, dtype=np.float32)
    initial_extrinsic_guess[:3, :3] = euler2mat(0.0, np.pi / 4.0, 0.0)
    initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.4], dtype=np.float32)
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)
    print(f"Initial extrinsic guess (OpenCV):\n{repr(initial_extrinsic_guess)}")

    # Paper mesh
    if args.paper_type not in PAPER_SIZES_M:
        raise ValueError(
            f"Unknown paper_type '{args.paper_type}'. Use 'letter' or 'a4'."
        )
    pw = PAPER_SIZES_M[args.paper_type]["width"]
    ph = PAPER_SIZES_M[args.paper_type]["height"]
    # Slightly thicker paper to avoid rasterization artifacts during visualization
    paper_box = trimesh.creation.box(extents=(pw, ph, 5e-3))

    # Load lerobot_sim2real/assets/robots/so101/meshes/base_so101_v2.stl
    base_so101_v2 = trimesh.load(
        "lerobot_sim2real/assets/robots/so101/meshes/base_so101_v2.stl"
    )

    meshes = [base_so101_v2]

    # World frame = paper pose, single link
    link_poses_dataset = np.eye(4, dtype=np.float32)[None, None, :, :]

    # Launch Gradio UI where user can generate mask and then run optimization
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_cfg = _normalize_sam2_cfg_name(args.model_cfg)
    annotator = PaperWebAnnotator(
        image=image,
        model_cfg=normalized_cfg,
        checkpoint=args.checkpoint,
        intrinsic_K=K,
        initial_extrinsic_opencv=initial_extrinsic_guess,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        camera_width=camera_width,
        camera_height=camera_height,
        train_steps=args.train_steps,
        early_stopping_steps=5000,
        output_dir=out_dir,
        paper_type=args.paper_type,
        server_name=args.server_name,
        server_port=args.server_port,
    )
    annotator.launch_and_wait()
    print(f"Session finished. Results (if optimization was run) saved to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(OpenCVPaperArgs))
