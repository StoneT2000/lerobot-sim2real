import os.path as osp
from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import trimesh

from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.utils import utils_3d


@dataclass
class RBSolverConfig:
    camera_height: int
    camera_width: int
    robot_masks: torch.Tensor
    link_poses_dataset: torch.Tensor
    meshes: List[Union[str, trimesh.Trimesh]]
    initial_extrinsic_guess: torch.Tensor


class RBSolver(nn.Module):
    """Rendering-based solver that optimizes camera extrinsic via mask overlap.

    IMPORTANT: When rendering, we use world-to-camera (view) transform. Given an
    internal parameterization for Tc_c2b (camera-to-base), we invert before
    composing with world (base) to link transforms for rendering.
    """

    def __init__(self, cfg: RBSolverConfig):
        super().__init__()
        self.cfg = cfg
        meshes = self.cfg.meshes
        for link_idx, mesh in enumerate(meshes):
            if isinstance(mesh, str):
                mesh = trimesh.load(osp.expanduser(mesh), force="mesh")
            else:
                assert isinstance(mesh, trimesh.Trimesh)
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f"vertices_{link_idx}", vertices)
            self.register_buffer(f"faces_{link_idx}", faces)
        self.nlinks = len(meshes)

        # camera parameters
        init_Tc_c2b = self.cfg.initial_extrinsic_guess
        init_dof = utils_3d.se3_log_map(
            torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1),
            eps=1e-5,
            backend="opencv",
        )[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)

        # renderer
        self.H, self.W = self.cfg.camera_height, self.cfg.camera_width
        self.renderer = NVDiffrastRenderer(self.H, self.W)
        self.register_buffer("history_ops", torch.zeros(10000, 6))

        # Unconditional debug
        try:
            init_T = torch.as_tensor(
                self.cfg.initial_extrinsic_guess, dtype=torch.float32
            )
            print(
                f"[RBSolver] init: H={self.H} W={self.W} links={self.nlinks}, init_t="
                f"[{float(init_T[0, 3]):.4f} {float(init_T[1, 3]):.4f} {float(init_T[2, 3]):.4f}]"
            )
        except Exception:
            print(f"[RBSolver] init: H={self.H} W={self.W} links={self.nlinks}")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        put_id = (self.history_ops == 0).all(dim=1).nonzero()[0, 0].item()
        self.history_ops[put_id] = self.dof.detach()

        # Internal parameterization produces Tc_c2b (camera-to-base)
        Tc_c2b = utils_3d.se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        # For rendering, we need world-to-camera (base->camera): Tw_w2c
        Tw_w2c = torch.linalg.inv(Tc_c2b)

        masks_ref: torch.Tensor = data["mask"].float()
        link_poses: torch.Tensor = data["link_poses"]
        intrinsic: torch.Tensor = data["intrinsic"].float()
        mount_poses: Optional[torch.Tensor] = data.get("mount_poses", None)

        assert link_poses.shape[0] == masks_ref.shape[0]
        assert link_poses.shape[1:] == (self.nlinks, 4, 4)
        assert masks_ref.shape[1:] == (self.H, self.W)

        batch_size = masks_ref.shape[0]
        # try:
        #     t_c2b = Tc_c2b[:3, 3].detach().cpu().numpy()
        #     t_w2c = Tw_w2c[:3, 3].detach().cpu().numpy()
        #     print(
        #         f"[RBSolver] forward: batch={batch_size}, "
        #         f"t_c2b=[{float(t_c2b[0]):.4f} {float(t_c2b[1]):.4f} {float(t_c2b[2]):.4f}], "
        #         f"t_w2c=[{float(t_w2c[0]):.4f} {float(t_w2c[1]):.4f} {float(t_w2c[2]):.4f}], "
        #         f"dof_trans=[{float(self.dof[0].item()):.4f} {float(self.dof[1].item()):.4f} {float(self.dof[2].item()):.4f}]"
        #     )
        # except Exception:
        #     print(f"[RBSolver] forward: batch={batch_size}")
        losses = []
        all_frame_all_link_si = []

        for bid in range(batch_size):
            all_link_si = []
            for link_idx in range(self.nlinks):
                if mount_poses is not None:
                    # Compose world->camera with world->mount->link
                    Tw2l = mount_poses[bid] @ link_poses[bid, link_idx]
                else:
                    Tw2l = link_poses[bid, link_idx]
                T_view_model = Tw_w2c @ Tw2l
                verts, faces = (
                    getattr(self, f"vertices_{link_idx}"),
                    getattr(self, f"faces_{link_idx}"),
                )
                si = self.renderer.render_mask(verts, faces, intrinsic, T_view_model)
                all_link_si.append(si)
            if len(all_link_si) == 1:
                all_link_si = all_link_si[0].reshape(1, self.H, self.W)
            else:
                all_link_si = torch.stack(all_link_si)
            all_link_si = all_link_si.sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            # Use mean squared error to stabilize scale across resolutions
            loss = torch.mean((all_link_si - masks_ref[bid]) ** 2)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)

        # try:
        #     pred0 = (all_frame_all_link_si[0] > 0.5).float()
        #     gt0 = (masks_ref[0] > 0.5).float()
        #     inter = (pred0 * gt0).sum()
        #     union = (pred0 + gt0).clamp(max=1).sum()
        #     iou0 = (inter / union).item() if union > 0 else 0.0
        #     print(
        #         f"[RBSolver] forward: loss={float(loss.item()):.6f}, iou0={float(iou0):.4f}"
        #     )
        # except Exception:
        #     print(f"[RBSolver] forward: loss={float(loss.item()):.6f}")
        output: Dict[str, Any] = {
            "rendered_masks": all_frame_all_link_si,
            "ref_masks": masks_ref,
            "error_maps": (all_frame_all_link_si - masks_ref).abs(),
            "mask_loss": loss,
        }

        if "gt_camera_pose" in data:
            gt_Tc_c2b = data["gt_camera_pose"]
            if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
                gt_dof6 = utils_3d.se3_log_map(
                    gt_Tc_c2b[None].permute(0, 2, 1), backend="opencv"
                )[0]
                trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180
                output["metrics"] = {
                    "err_x": trans_err[0].item(),
                    "err_y": trans_err[1].item(),
                    "err_z": trans_err[2].item(),
                    "err_trans": trans_err.norm().item(),
                    "err_rot": rot_err.item(),
                }

        return output

    def get_predicted_extrinsic(self) -> torch.Tensor:
        # Return Tc_c2b to keep external API consistent
        return utils_3d.se3_exp_map(self.dof[None].detach()).permute(0, 2, 1)[0]
