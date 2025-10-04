"""Unit tests for urdf_utils.py - mesh extraction functions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import trimesh

from lerobot_sim2real.utils.urdf_utils import (
    extract_individual_meshes_with_origins,
    extract_merged_meshes_per_link,
)


class TestExtractIndividualMeshesWithOrigins:
    """Test extract_individual_meshes_with_origins function."""

    def test_extract_no_meshes(self):
        """Test extraction with URDF containing no visual meshes."""
        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.visuals = []
        mock_urdf.links = [mock_link]

        meshes, names, origins = extract_individual_meshes_with_origins(mock_urdf)

        assert len(meshes) == 0
        assert len(names) == 0
        assert len(origins) == 0

    def test_extract_single_mesh(self):
        """Test extraction with single mesh."""
        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        # Create mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.faces = np.array([[0, 1, 2]])

        # Create mock visual with mesh
        mock_visual = Mock()
        mock_geometry = Mock()
        mock_mesh_attr = Mock()
        mock_mesh_attr.meshes = [mock_mesh]
        mock_geometry.mesh = mock_mesh_attr
        mock_visual.geometry = mock_geometry
        mock_visual.origin = np.eye(4)

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names, origins = extract_individual_meshes_with_origins(mock_urdf)

        assert len(meshes) == 1
        assert len(names) == 1
        assert len(origins) == 1
        assert names[0] == "test_link"
        assert np.array_equal(origins[0], np.eye(4))

    def test_extract_multiple_visuals_per_link(self):
        """Test extraction with multiple visual elements per link."""
        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "multi_link"

        # Create two mock meshes
        mock_mesh1 = Mock()
        mock_mesh1.vertices = np.array([[0, 0, 0]])
        mock_mesh1.faces = np.array([[0, 0, 0]])

        mock_mesh2 = Mock()
        mock_mesh2.vertices = np.array([[1, 1, 1]])
        mock_mesh2.faces = np.array([[0, 0, 0]])

        # Create two visuals
        mock_visual1 = Mock()
        mock_geometry1 = Mock()
        mock_mesh_attr1 = Mock()
        mock_mesh_attr1.meshes = [mock_mesh1]
        mock_geometry1.mesh = mock_mesh_attr1
        mock_visual1.geometry = mock_geometry1
        mock_visual1.origin = np.eye(4)

        mock_visual2 = Mock()
        mock_geometry2 = Mock()
        mock_mesh_attr2 = Mock()
        mock_mesh_attr2.meshes = [mock_mesh2]
        mock_geometry2.mesh = mock_mesh_attr2
        mock_visual2.geometry = mock_geometry2
        mock_visual2.origin = np.eye(4) * 2

        mock_link.visuals = [mock_visual1, mock_visual2]
        mock_urdf.links = [mock_link]

        meshes, names, origins = extract_individual_meshes_with_origins(mock_urdf)

        assert len(meshes) == 2
        assert all(name == "multi_link" for name in names)

    def test_extract_skip_non_mesh_geometry(self):
        """Test that non-mesh geometry is skipped."""
        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        # Create visual without mesh (e.g., cylinder, box)
        mock_visual = Mock()
        mock_geometry = Mock()
        mock_geometry.mesh = None  # No mesh attribute
        mock_visual.geometry = mock_geometry

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names, origins = extract_individual_meshes_with_origins(mock_urdf)

        assert len(meshes) == 0

    def test_extract_visual_origin_transformation(self):
        """Test that visual origins are correctly extracted."""
        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0]])
        mock_mesh.faces = np.array([[0, 0, 0]])

        mock_visual = Mock()
        mock_geometry = Mock()
        mock_mesh_attr = Mock()
        mock_mesh_attr.meshes = [mock_mesh]
        mock_geometry.mesh = mock_mesh_attr
        mock_visual.geometry = mock_geometry

        # Custom visual origin
        custom_origin = np.eye(4)
        custom_origin[:3, 3] = [1, 2, 3]
        mock_visual.origin = custom_origin

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names, origins = extract_individual_meshes_with_origins(mock_urdf)

        assert len(origins) == 1
        assert np.array_equal(origins[0], custom_origin)


class TestExtractMergedMeshesPerLink:
    """Test extract_merged_meshes_per_link function."""

    @patch("lerobot_sim2real.utils.urdf_utils.merge_meshes")
    def test_extract_and_merge(self, mock_merge):
        """Test extraction and merging of meshes."""
        # Setup mock merged mesh
        merged_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=np.array([[0, 1, 2]]),
        )
        mock_merge.return_value = merged_mesh

        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        # Create mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.faces = np.array([[0, 1, 2]])

        mock_visual = Mock()
        mock_geometry = Mock()
        mock_mesh_attr = Mock()
        mock_mesh_attr.meshes = [mock_mesh]
        mock_geometry.mesh = mock_mesh_attr
        mock_visual.geometry = mock_geometry
        mock_visual.origin = np.eye(4)

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names = extract_merged_meshes_per_link(mock_urdf)

        assert len(meshes) == 1
        assert len(names) == 1
        assert names[0] == "test_link"
        mock_merge.assert_called_once()

    @patch("lerobot_sim2real.utils.urdf_utils.merge_meshes")
    def test_skip_when_merge_returns_none(self, mock_merge):
        """Test that links are skipped when merge returns None."""
        mock_merge.return_value = None

        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0]])
        mock_mesh.faces = np.array([[0, 0, 0]])

        mock_visual = Mock()
        mock_geometry = Mock()
        mock_mesh_attr = Mock()
        mock_mesh_attr.meshes = [mock_mesh]
        mock_geometry.mesh = mock_mesh_attr
        mock_visual.geometry = mock_geometry
        mock_visual.origin = np.eye(4)

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names = extract_merged_meshes_per_link(mock_urdf)

        assert len(meshes) == 0
        assert len(names) == 0

    @patch("lerobot_sim2real.utils.urdf_utils.merge_meshes")
    def test_visual_origin_applied_before_merge(self, mock_merge):
        """Test that visual origins are applied to meshes before merging."""
        merged_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0]]), faces=np.array([[0, 0, 0]])
        )
        mock_merge.return_value = merged_mesh

        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        # Original mesh vertices
        original_vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_mesh = Mock()
        mock_mesh.vertices = original_vertices
        mock_mesh.faces = np.array([[0, 1, 2]])

        mock_visual = Mock()
        mock_geometry = Mock()
        mock_mesh_attr = Mock()
        mock_mesh_attr.meshes = [mock_mesh]
        mock_geometry.mesh = mock_mesh_attr
        mock_visual.geometry = mock_geometry

        # Visual origin with translation
        visual_origin = np.eye(4)
        visual_origin[:3, 3] = [10, 20, 30]
        mock_visual.origin = visual_origin

        mock_link.visuals = [mock_visual]
        mock_urdf.links = [mock_link]

        meshes, names = extract_merged_meshes_per_link(mock_urdf)

        # Verify merge was called with transformed mesh
        mock_merge.assert_called_once()
        transformed_meshes = mock_merge.call_args[0][0]
        assert len(transformed_meshes) == 1

        # Check that vertices were transformed
        transformed_mesh = transformed_meshes[0]
        # The transformed vertices should have the translation applied
        expected_vertices = original_vertices + np.array([10, 20, 30])
        np.testing.assert_array_almost_equal(
            transformed_mesh.vertices, expected_vertices
        )

    @patch("lerobot_sim2real.utils.urdf_utils.merge_meshes")
    def test_multiple_visuals_merged_per_link(self, mock_merge):
        """Test that multiple visual elements are merged into one mesh per link."""
        merged_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0]]), faces=np.array([[0, 0, 0]])
        )
        mock_merge.return_value = merged_mesh

        mock_urdf = Mock()
        mock_link = Mock()
        mock_link.name = "test_link"

        # Create two meshes
        mock_mesh1 = Mock()
        mock_mesh1.vertices = np.array([[0, 0, 0]])
        mock_mesh1.faces = np.array([[0, 0, 0]])

        mock_mesh2 = Mock()
        mock_mesh2.vertices = np.array([[1, 1, 1]])
        mock_mesh2.faces = np.array([[0, 0, 0]])

        # Create two visuals
        mock_visual1 = Mock()
        mock_geometry1 = Mock()
        mock_mesh_attr1 = Mock()
        mock_mesh_attr1.meshes = [mock_mesh1]
        mock_geometry1.mesh = mock_mesh_attr1
        mock_visual1.geometry = mock_geometry1
        mock_visual1.origin = np.eye(4)

        mock_visual2 = Mock()
        mock_geometry2 = Mock()
        mock_mesh_attr2 = Mock()
        mock_mesh_attr2.meshes = [mock_mesh2]
        mock_geometry2.mesh = mock_mesh_attr2
        mock_visual2.geometry = mock_geometry2
        mock_visual2.origin = np.eye(4)

        mock_link.visuals = [mock_visual1, mock_visual2]
        mock_urdf.links = [mock_link]

        meshes, names = extract_merged_meshes_per_link(mock_urdf)

        # Should result in one merged mesh per link
        assert len(meshes) == 1
        assert len(names) == 1

        # Verify that merge was called with both transformed meshes
        mock_merge.assert_called_once()
        transformed_meshes = mock_merge.call_args[0][0]
        assert len(transformed_meshes) == 2
