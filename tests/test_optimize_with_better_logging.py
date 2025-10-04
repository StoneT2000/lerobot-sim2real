"""Unit tests for optimize_with_better_logging.py."""

import pytest
import numpy as np
import torch
import trimesh
from unittest.mock import patch

from lerobot_sim2real.optim.optimize_with_better_logging import optimize


@pytest.fixture
def mock_mesh():
    """Create a simple mock trimesh object."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


@pytest.fixture
def optimization_inputs(mock_mesh):
    """Create basic inputs for the optimize function."""
    camera_height = 480
    camera_width = 640

    initial_extrinsic = torch.eye(4)
    initial_extrinsic[:3, 3] = torch.tensor([0.5, 0.0, 0.3])

    camera_intrinsic = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )

    num_samples = 10
    num_links = 2
    masks = torch.ones((num_samples, camera_height, camera_width))
    link_poses_dataset = (
        torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(num_samples, num_links, 1, 1)
    )

    meshes = [mock_mesh, mock_mesh]

    return {
        "initial_extrinsic_guess": initial_extrinsic,
        "camera_intrinsic": camera_intrinsic,
        "masks": masks,
        "link_poses_dataset": link_poses_dataset,
        "meshes": meshes,
        "camera_width": camera_width,
        "camera_height": camera_height,
    }


class MockRBSolverInstance:
    """Mock RBSolver instance that behaves like the real one."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.call_count = 0
        self.param = torch.nn.Parameter(torch.randn(6), requires_grad=True)

    def forward(self, data):
        return self.__call__(data)

    def __call__(self, data):
        batch_size = data["mask"].shape[0]
        H, W = data["mask"].shape[1], data["mask"].shape[2]

        # Loss decreases over time to simulate optimization
        self.call_count += 1
        base_loss = 0.1
        loss_value = base_loss * (1.0 - self.call_count * 0.001)
        loss_value = max(loss_value, 0.001)  # Don't go below this
        loss = torch.tensor(loss_value, requires_grad=True)

        output = {
            "rendered_masks": torch.zeros((batch_size, H, W), requires_grad=True),
            "ref_masks": data["mask"],
            "error_maps": torch.zeros((batch_size, H, W)),
            "mask_loss": loss,
        }

        if "gt_camera_pose" in data and data["gt_camera_pose"] is not None:
            output["metrics"] = {
                "err_x": 0.01,
                "err_y": 0.02,
                "err_z": 0.03,
                "err_trans": 0.04,
                "err_rot": 0.5,
            }

        return output

    def get_predicted_extrinsic(self):
        extrinsic = torch.eye(4)
        extrinsic[:3, 3] = torch.tensor([0.51, 0.01, 0.31])
        return extrinsic

    def parameters(self):
        return [self.param]

    def to(self, device):
        self.param = self.param.to(device)
        return self


@pytest.fixture
def mock_rb_solver():
    """Mock RBSolver to avoid GPU dependencies."""
    with patch(
        "lerobot_sim2real.optim.optimize_with_better_logging.RBSolver",
        MockRBSolverInstance,
    ):
        yield


def test_optimize_basic(optimization_inputs, mock_rb_solver):
    """Test basic optimization workflow."""
    result = optimize(
        **optimization_inputs,
        iterations=10,
        learning_rate=1e-3,
        verbose=False,
    )

    # Should return a 4x4 extrinsic matrix
    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("batch_size", [None, 4])
def test_optimize_with_batching(optimization_inputs, mock_rb_solver, batch_size):
    """Test optimization with and without batch processing."""
    result = optimize(
        **optimization_inputs,
        iterations=20,
        learning_rate=1e-3,
        batch_size=batch_size,
        verbose=False,
    )

    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()


def test_optimize_early_stopping(optimization_inputs):
    """Test that early stopping works when loss doesn't improve."""

    class ConstantLossMockSolver:
        def __init__(self, cfg):
            self.cfg = cfg
            self.param = torch.nn.Parameter(torch.randn(6), requires_grad=True)

        def __call__(self, data):
            batch_size = data["mask"].shape[0]
            H, W = data["mask"].shape[1], data["mask"].shape[2]
            loss = torch.tensor(0.1, requires_grad=True)
            return {
                "rendered_masks": torch.zeros((batch_size, H, W), requires_grad=True),
                "ref_masks": data["mask"],
                "error_maps": torch.zeros((batch_size, H, W)),
                "mask_loss": loss,
            }

        def get_predicted_extrinsic(self):
            return torch.eye(4)

        def parameters(self):
            return [self.param]

        def to(self, device):
            return self

    with patch(
        "lerobot_sim2real.optim.optimize_with_better_logging.RBSolver",
        ConstantLossMockSolver,
    ):
        result = optimize(
            **optimization_inputs,
            iterations=1000,
            learning_rate=1e-3,
            early_stopping_steps=10,
            verbose=False,
        )

        # Should still return a valid result
        assert result.shape == (4, 4)


def test_optimize_return_history(optimization_inputs, mock_rb_solver):
    """Test that return_history returns proper data structure."""
    result = optimize(
        **optimization_inputs,
        iterations=20,
        learning_rate=1e-3,
        verbose=False,
        return_history=True,
    )

    # Should return a dictionary when return_history=True
    assert isinstance(result, dict)
    assert "best_extrinsics" in result
    assert "best_extrinsics_step" in result
    assert "best_extrinsics_losses" in result
    assert "all_losses" in result

    # Check types and shapes
    assert isinstance(result["best_extrinsics"], torch.Tensor)
    assert result["best_extrinsics"].shape[1:] == (4, 4)

    # Best losses should be monotonically decreasing
    best_losses = result["best_extrinsics_losses"]
    for i in range(1, len(best_losses)):
        assert best_losses[i] <= best_losses[i - 1]


def test_optimize_with_ground_truth(optimization_inputs, mock_rb_solver):
    """Test optimization with ground truth pose for metrics."""
    gt_pose = torch.eye(4)
    gt_pose[:3, 3] = torch.tensor([0.52, 0.01, 0.31])

    result = optimize(
        **optimization_inputs,
        iterations=10,
        learning_rate=1e-3,
        gt_camera_pose=gt_pose,
        verbose=True,  # Needed for pbar.set_postfix
    )

    assert result.shape == (4, 4)


def test_optimize_with_mount_poses(optimization_inputs, mock_rb_solver):
    """Test optimization with moving camera mount."""
    num_samples = optimization_inputs["masks"].shape[0]
    mount_poses = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
    mount_poses[:, :3, 3] = torch.randn(num_samples, 3) * 0.1

    result = optimize(
        **optimization_inputs,
        camera_mount_poses=mount_poses,
        iterations=10,
        learning_rate=1e-3,
        verbose=False,
    )

    assert result.shape == (4, 4)


def test_optimize_integration(optimization_inputs, mock_rb_solver):
    """Test optimization with multiple features combined."""
    num_samples = optimization_inputs["masks"].shape[0]
    mount_poses = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
    gt_pose = torch.eye(4)
    gt_pose[:3, 3] = torch.tensor([0.52, 0.01, 0.31])

    result = optimize(
        **optimization_inputs,
        camera_mount_poses=mount_poses,
        iterations=30,
        learning_rate=1e-3,
        gt_camera_pose=gt_pose,
        batch_size=None,
        early_stopping_steps=20,
        verbose=True,
        return_history=True,
    )

    # Check result structure
    assert isinstance(result, dict)
    assert "best_extrinsics" in result
    assert result["best_extrinsics"].shape[1:] == (4, 4)
    assert len(result["all_losses"]) > 0
