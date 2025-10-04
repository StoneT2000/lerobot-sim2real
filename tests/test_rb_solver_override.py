"""Unit tests for rb_solver_override.py."""

import pytest
import numpy as np
import torch
import trimesh
from unittest.mock import patch

from lerobot_sim2real.optim.rb_solver_override import RBSolver, RBSolverConfig


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
def solver_config(mock_mesh):
    """Create a basic RBSolverConfig for testing."""
    camera_height = 480
    camera_width = 640

    # Create dummy data
    robot_masks = torch.ones((4, camera_height, camera_width))
    link_poses_dataset = torch.eye(4).unsqueeze(0).repeat(4, 2, 1, 1)

    # Initial camera extrinsic (camera-to-base transform)
    initial_extrinsic = torch.eye(4)
    initial_extrinsic[:3, 3] = torch.tensor([0.5, 0.0, 0.3])  # translation

    meshes = [mock_mesh, mock_mesh]  # Two links

    cfg = RBSolverConfig(
        camera_height=camera_height,
        camera_width=camera_width,
        robot_masks=robot_masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        initial_extrinsic_guess=initial_extrinsic,
    )

    return cfg


@pytest.fixture
def mock_renderer():
    """Mock the NVDiffrastRenderer to avoid GPU dependencies in tests."""
    with patch(
        "lerobot_sim2real.optim.rb_solver_override.NVDiffrastRenderer"
    ) as MockRenderer:
        # Keep track of the renderer dimensions
        class MockRendererInstance:
            def __init__(self, H, W):
                self.H = H
                self.W = W

            def render_mask(self, *args, **kwargs):
                # Return a mask with the correct size and gradient support
                return torch.zeros((self.H, self.W), requires_grad=True)

        # Replace the constructor to return our mock instance
        MockRenderer.side_effect = MockRendererInstance
        yield MockRenderer


def test_config_creation(solver_config):
    """Test that RBSolverConfig can be created with valid parameters."""
    assert solver_config.camera_height == 480
    assert solver_config.camera_width == 640
    assert solver_config.robot_masks.shape == (4, 480, 640)
    assert solver_config.link_poses_dataset.shape == (4, 2, 4, 4)
    assert len(solver_config.meshes) == 2
    assert solver_config.initial_extrinsic_guess.shape == (4, 4)


def test_solver_initialization(solver_config, mock_renderer):
    """Test RBSolver initialization."""
    solver = RBSolver(solver_config)

    assert solver.H == 480
    assert solver.W == 640
    assert solver.nlinks == 2
    assert hasattr(solver, "dof")
    assert solver.dof.requires_grad is True
    assert solver.dof.shape == (6,)  # 6-DOF (3 translation + 3 rotation)

    # Check that mesh vertices and faces are registered as buffers
    assert hasattr(solver, "vertices_0")
    assert hasattr(solver, "faces_0")
    assert hasattr(solver, "vertices_1")
    assert hasattr(solver, "faces_1")


def test_forward_basic(solver_config, mock_renderer):
    """Test basic forward pass."""
    solver = RBSolver(solver_config)

    batch_size = 2
    data = {
        "mask": torch.ones((batch_size, 480, 640)),
        "link_poses": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, 2, 1, 1),
        "intrinsic": torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
    }

    output = solver.forward(data)

    # Check output structure
    assert "rendered_masks" in output
    assert "ref_masks" in output
    assert "error_maps" in output
    assert "mask_loss" in output

    # Check shapes
    assert output["rendered_masks"].shape == (batch_size, 480, 640)
    assert output["ref_masks"].shape == (batch_size, 480, 640)
    assert output["mask_loss"].shape == ()  # scalar

    # Check that loss is finite and non-negative
    assert torch.isfinite(output["mask_loss"])
    assert output["mask_loss"] >= 0


def test_forward_with_mount_poses(solver_config, mock_renderer):
    """Test forward pass with mount poses."""
    solver = RBSolver(solver_config)

    batch_size = 2
    mount_pose = torch.eye(4)
    mount_pose[:3, 3] = torch.tensor([0.1, 0.2, 0.3])

    data = {
        "mask": torch.ones((batch_size, 480, 640)),
        "link_poses": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, 2, 1, 1),
        "intrinsic": torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "mount_poses": mount_pose.unsqueeze(0).repeat(batch_size, 1, 1),
    }

    output = solver.forward(data)

    assert output["mask_loss"] >= 0
    assert torch.isfinite(output["mask_loss"])


def test_forward_with_gt_camera_pose(solver_config, mock_renderer):
    """Test forward pass with ground truth camera pose for metrics."""
    solver = RBSolver(solver_config)

    batch_size = 1
    gt_camera_pose = torch.eye(4)
    gt_camera_pose[:3, 3] = torch.tensor([0.52, 0.01, 0.31])

    data = {
        "mask": torch.ones((batch_size, 480, 640)),
        "link_poses": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, 2, 1, 1),
        "intrinsic": torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "gt_camera_pose": gt_camera_pose,
    }

    output = solver.forward(data)

    # When ground truth is provided, metrics should be in output
    assert "metrics" in output
    assert "err_x" in output["metrics"]
    assert "err_y" in output["metrics"]
    assert "err_z" in output["metrics"]
    assert "err_trans" in output["metrics"]
    assert "err_rot" in output["metrics"]


def test_get_predicted_extrinsic(solver_config, mock_renderer):
    """Test getting the predicted camera extrinsic."""
    solver = RBSolver(solver_config)

    extrinsic = solver.get_predicted_extrinsic()

    # Should be a 4x4 transformation matrix
    assert extrinsic.shape == (4, 4)

    # Bottom row should be [0, 0, 0, 1]
    assert torch.allclose(
        extrinsic[3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5
    )

    # Should be detached (no gradients)
    assert extrinsic.requires_grad is False


def test_gradient_flow(solver_config, mock_renderer):
    """Test that the solver parameters are set up for gradient computation."""
    solver = RBSolver(solver_config)

    # Check that the dof parameter is set up correctly for gradients
    assert solver.dof.requires_grad is True
    assert solver.dof.shape == (6,)

    # Test that we can compute a simple loss and backprop through dof
    simple_loss = (solver.dof**2).sum()
    simple_loss.backward()

    # Check that dof parameter has gradients
    assert solver.dof.grad is not None
    assert torch.isfinite(solver.dof.grad).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size, solver_config, mock_renderer):
    """Test solver with different batch sizes."""
    solver = RBSolver(solver_config)

    data = {
        "mask": torch.ones((batch_size, 480, 640)),
        "link_poses": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, 2, 1, 1),
        "intrinsic": torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
    }

    output = solver.forward(data)

    assert output["rendered_masks"].shape == (batch_size, 480, 640)
    assert output["ref_masks"].shape == (batch_size, 480, 640)
