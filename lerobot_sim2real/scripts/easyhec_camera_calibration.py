"""Code modified from https://github.com/stonet2000/simple-easyhec/blob/main/easyhec/examples/real/so100.py"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from lerobot.cameras.realsense import RealSenseCamera

from lerobot.motors.motors_bus import MotorNormMode
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from transforms3d.euler import euler2mat
from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv

# Import shared URDF utilities
from lerobot_sim2real.utils.urdf_utils import load_robot_meshes_for_calibration

# Import our custom visualization function
from lerobot_sim2real.utils.custom_visualization import (
    visualize_extrinsic_results_red_mask,
)
# from easyhec import ROBOT_DEFINITIONS_DIR

from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.utils.camera import scale_intrinsics


@dataclass
class SO101Args(Args):
    """Calibrate a (realsense) camera with LeRobot SO101. Note that this script might not work with your particular realsense camera, modify as needed. Other cameras can work if you modify the code to get the camera intrinsics and a single color image from the camera. Results are saved to {output_dir} and organized by the camera name specified in the robot config. Currently only supports off-hand cameras

    For your own usage you may have a different camera setup, robot, calibration offsets etc., so we recommend you to copy this file at https://github.com/stonet2000/simple-easyhec/blob/main/easyhec/examples/real/so101.py.

    Before usage make sure to calibrate the robot's motors according to the LeRobot tutorial and look for all comments that start with "CHECK:" which highlight the following:

    1. Check the robot config and make sure the correct camera is used. The default script is for a single realsense camera labelled as "base_camera".
    2. Check and modify the CALIBRATION_OFFSET dictionary to match your own robot's calibration offsets. This is extremely important to tune and is necessary since the 0 degree position of the joints in the real world when calibrated with LeRobot currently do not match the 0 degree position when rendered/simulated.
    3. Modify the initial extrinsic guess if the optimization process fails to converge to a good solution. To save time you can also turn on --use-previous-captures to skip the data collection process if already done once.

    Note that LeRobot SO101 motor calibration is done by moving most joints from one end to another. Make sure to move the joints are far as possible during the LeRobot tutorial on caibration for best results.

    """

    output_dir: str = "results/so101"
    use_previous_captures: bool = False
    """If True, will use the previous collected images and robot segmentations if they exist which can save you time. Otherwise, will prompt you to generate a new segmentation mask. This is useful if you find the initial extrinsic guess is not good enough and simply want to refine that and want to skip the segmentation process."""

    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""

    auto_mask: bool = False
    """If True, skip interactive UI and auto-generate SAM2 masks headlessly."""


def main(args: SO101Args):
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

    user_tuned_calibration_offset = False
    for k in CALIBRATION_OFFSET.keys():
        if CALIBRATION_OFFSET[k] != 0:
            user_tuned_calibration_offset = True
            break
    if not user_tuned_calibration_offset:
        logging.warning(
            "The calibration offset for sim2real/real2sim is not tuned!! Unless you are absolutely sure you will most likely get poor results."
        )

    robot: SO101Follower = create_real_robot("so101")
    robot_id = robot.id if robot.id is not None else "default"
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    robot.connect()

    # Use the public cameras dict for consistent naming across intrinsics/images/extrinsics
    cameras_ft = robot.cameras
    print(f"Found {len(cameras_ft)} cameras to calibrate")
    for k in cameras_ft.keys():
        (Path(args.output_dir) / robot_id / k).mkdir(parents=True, exist_ok=True)

    ### Make an initial guess for the extrinsic for each camera ###
    # CHECK: Double check this initial extrinsic guess is roughly close to the real world.
    initial_extrinsic_guesses = dict()
    for k in cameras_ft.keys():
        initial_extrinsic_guess = np.eye(4)

        # the guess says we are at position xyz=[-0.4, 0.0, 0.4] and angle the camera downwards by np.pi / 4 radians  or 45 degrees
        # note that this convention is more natural for robotics (follows the typical convention for ROS and various simulators), where +Z is moving up towards the sky, +Y is to the left, +X is forward
        initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
        initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.5])
        initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

        initial_extrinsic_guesses[k] = initial_extrinsic_guess

    print("Initial extrinsic guesses")
    for k in initial_extrinsic_guesses.keys():
        print(f"Camera {k}:\n{repr(initial_extrinsic_guesses[k])}")

    # get camera intrinsics
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
            # Fallback: derive a pinhole intrinsic assuming square pixels and principal point at image center
            try:
                frame = cam.async_read()
                height, width = frame.shape[:2]
                fx = fy = 0.8 * max(width, height)
                cx = width / 2.0
                cy = height / 2.0
                intrinsics[cam_name] = np.array(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ]
                )
            except Exception:
                # If we can't read a frame now, we'll skip this camera later
                pass

    ### Data Collection Process below ###
    # We move the robot to a few joint configurations and collect images and generate a link pose dataset.

    joint_position_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    # Get current positions in degrees, convert to radians
    # then return a np vector of the positions
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

    # Use shared utility function to load URDF and extract meshes for calibration
    robot_urdf, meshes, mesh_link_names = load_robot_meshes_for_calibration("so101")

    if (
        args.use_previous_captures
        and (Path(args.output_dir) / robot_id / "link_poses_dataset.npy").exists()
    ):
        # load the previous captures
        link_poses_dataset = np.load(
            Path(args.output_dir) / robot_id / "link_poses_dataset.npy"
        )
        image_dataset = np.load(
            Path(args.output_dir) / robot_id / "image_dataset.npy", allow_pickle=True
        ).reshape(-1)[0]
    else:
        # reference qpos positions to calibrate with
        qpos_samples = [
            np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0.2]),
            np.array([np.pi / 3, -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0]),
        ]
        control_freq = 15
        max_radians_per_step = 0.05

        # generate our link pose dataset and image pairs. We do this by moving the robot to the reference joint positions and collecting images from all cameras
        link_poses_dataset = np.zeros((len(qpos_samples), len(meshes), 4, 4))
        image_dataset = defaultdict(list)

        for i in range(len(qpos_samples)):
            # control code for lerobot below
            goal_qpos = qpos_samples[i]
            target_qpos = get_qpos(robot)
            for _ in range(int(20 * control_freq)):
                start_loop_t = time.perf_counter()
                delta_qpos = goal_qpos - target_qpos
                delta_step = delta_qpos.clip(
                    min=-max_radians_per_step, max=max_radians_per_step
                )

                # Step until target and goal are within some threshold distance/norm
                if np.linalg.norm(delta_qpos) < 1e-4:
                    break
                target_qpos += delta_step
                dt_s = time.perf_counter() - start_loop_t

                # Move the robot to the new target
                set_target_qpos(robot, target_qpos)
                time.sleep(1 / control_freq - dt_s)
            time.sleep(
                1
            )  # give some time for the robot to settle, cheap arms don't hold up as well
            qpos_dict = get_qpos(robot, flat=False)
            for cam_name, cam in robot.cameras.items():
                image_dataset[cam_name].append(cam.async_read())

            # get link poses
            # Build cfg including only available joints; default others (e.g., fixed joints) to 0
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

    ### Camera Calibration Process below ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize SAM2 config path to Hydra-compatible names expected by build_sam2:
    # - Prefer 'configs/sam2.1/sam2.1_hiera_l.yaml'
    # - Accept full filesystem paths, short 'sam2.1/sam2.1_hiera_l', or just 'sam2.1_hiera_l'
    def _normalize_sam2_cfg_name(cfg: str) -> str:
        s = str(cfg)
        if "configs/" in s:
            # Ensure we pass from 'configs/...' onward, preserving extension
            return s[s.index("configs/") :]
        # If user passed a short hydra-ish path like 'sam2.1/sam2.1_hiera_l'
        if "/" in s:
            base = s
            if not (base.endswith(".yaml") or base.endswith(".yml")):
                base = f"{base}.yaml"
            return f"configs/{base}"
        # If only a bare name like 'sam2.1_hiera_l'
        name = s
        if not (name.endswith(".yaml") or name.endswith(".yml")):
            name = f"{name}.yaml"
        # Default to sam2.1; if user intended sam2, they should pass folder explicitly
        return f"configs/sam2.1/{name}"

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
        camera_mount_poses = None  # TODO (stao): support this
        camera_width = images.shape[2]
        camera_height = images.shape[1]

        mask_path = Path(args.output_dir) / robot_id / k / "mask.npy"
        if args.use_previous_captures and mask_path.exists():
            print(f"Using previous mask from {mask_path}")
            masks = np.load(mask_path)
        else:
            normalized_cfg = _normalize_sam2_cfg_name(args.model_cfg)
            print(f"Using SAM2 model_cfg: {normalized_cfg}")
            if args.auto_mask:
                # Headless: build predictor and generate masks with synthetic prompts
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                device_str = "cuda" if torch.cuda.is_available() else "cpu"
                predictor = SAM2ImagePredictor(
                    build_sam2(normalized_cfg, args.checkpoint, device=device_str)
                )
                gen_masks = []
                for img in images:
                    h, w = img.shape[:2]
                    # Positive point at center; negatives at corners to reduce bleed
                    input_point = np.array(
                        [
                            [w / 2.0, h / 2.0],
                            [0.05 * w, 0.05 * h],
                            [0.95 * w, 0.05 * h],
                            [0.05 * w, 0.95 * h],
                            [0.95 * w, 0.95 * h],
                        ]
                    ).astype(np.float32)
                    input_label = np.array([1, -1, -1, -1, -1]).astype(np.int32)
                    with torch.inference_mode():
                        predictor.set_image(img)
                        mask, _, _ = predictor.predict(
                            input_point, input_label, multimask_output=False
                        )
                    gen_masks.append(mask[0])
                masks = np.stack(gen_masks)
                np.save(mask_path, masks)
            else:
                # Interactive UI (requires GUI)
                interactive_segmentation = InteractiveSegmentation(
                    segmentation_model="sam2",
                    segmentation_model_cfg=dict(
                        checkpoint=args.checkpoint,
                        model_cfg=normalized_cfg,
                    ),
                )
                masks = interactive_segmentation.get_segmentation(images)
                np.save(mask_path, masks)

        ### run the optimization given the data ###
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

        ### Print predicted results ###

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
            mask_color=(255, 0, 0),  # Red color
        )
        print(f"Visualizations saved to {Path(args.output_dir) / robot_id / k}")


if __name__ == "__main__":
    main(tyro.cli(SO101Args))
