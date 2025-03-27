# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from neural_wbc.core import ReferenceMotionState

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class RefMotionVisualizer:
    """Visualizer of the reference motions."""

    def __init__(self):
        self._initialized = False
        self._active_ref_motion_markers = None
        self._inactive_ref_motion_markers = None

    def _initialize_ref_motion_markers(self):
        print("Initialize markers for ref motion joints.")
        # Visualizer for the active reference body positions.
        active_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        active_marker_cfg.markers["target"].radius = 0.05
        active_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )  # blue
        active_marker_cfg.prim_path = "/Visuals/Command/active_ref_motion"
        self._active_ref_motion_markers = VisualizationMarkers(active_marker_cfg)
        self._active_ref_motion_markers.set_visibility(True)

        # Visualizere for the inactive reference body positions.
        inactive_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        inactive_marker_cfg.markers["target"].radius = 0.03
        inactive_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )  # red
        inactive_marker_cfg.prim_path = "/Visuals/Command/inactive_ref_motion"
        self._inactive_ref_motion_markers = VisualizationMarkers(inactive_marker_cfg)
        self._inactive_ref_motion_markers.set_visibility(True)

        # Visualizere for the reference joint angles.
        joint_angle_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        joint_angle_marker_cfg.markers["target"].radius = 0.03
        joint_angle_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)
        )  # green
        joint_angle_marker_cfg.prim_path = "/Visuals/Command/ref_joint_angle"
        self.ref_joint_angle_markers = VisualizationMarkers(joint_angle_marker_cfg)
        self.ref_joint_angle_markers.set_visibility(True)

        # Visualize for the root arrow
        root_marker_cfg = VisualizationMarkersCfg()
        root_marker_cfg.prim_path = "/Visuals/Command/root_direction"
        root_marker_cfg.markers = {
            "arrow_x": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            }
        self.root_markers = VisualizationMarkers(root_marker_cfg)
        self.root_markers.set_visibility(True)

        self._initialized = True

    def visualize(self, ref_motion: ReferenceMotionState, mask: torch.Tensor | None = None):
        if not self._initialized:
            self._initialize_ref_motion_markers()

        # Split the active and inactive body references.
        if mask is None:
            active_body_pos = ref_motion.body_pos_extend.view(-1, 3)
            device = active_body_pos.device
            inactive_body_pos = torch.zeros(0, 3, device=device)
            active_joint_angle_pos = torch.zeros(0, 1, device=device)
            root_pos = torch.zeros(0, 4, device=device)
        else:
            num_bodies = ref_motion.body_pos_extend.shape[1]
            mask_keypoints = mask[:, :num_bodies]
            # We reshape to still have the correct shape even if the mask is set to all false.
            # Else we will try to visualize non-existing bodies below.
            mask_keypoints_flat = mask_keypoints.flatten()
            active_body_pos = ref_motion.body_pos_extend.view(-1, 3)[mask_keypoints_flat, :]
            inactive_body_pos = ref_motion.body_pos_extend.view(-1, 3)[~mask_keypoints_flat, :]

            mask_joint_angles = torch.cat([torch.zeros((mask.shape[0], 1), device=mask.device).to(bool), mask[:, num_bodies:-7]], dim=1)

            mask_joint_angles_flat = mask_joint_angles.flatten()
            active_joint_angle_pos = ref_motion.body_pos.view(-1, 3)[mask_joint_angles_flat, :]
            active_joint_angle_pos[:, 2] += 0.01

            active_root = torch.where(mask[:, -7:].sum(dim=1) > 0)[0]
            root_pos = ref_motion.root_pos.view(-1, 3)[active_root, :]
            root_pos[:, 2] += 1.0
            root_rot = ref_motion.root_rot.view(-1, 4)[active_root, :]

        # Update the position of the visualization markers.
        if active_body_pos.shape[0] != 0:
            # Need to set visibility to True to ensure the markers are visible.
            self._active_ref_motion_markers.set_visibility(True)
            self._active_ref_motion_markers.visualize(active_body_pos)
        else:
            self._active_ref_motion_markers.set_visibility(False)

        if inactive_body_pos.shape[0] != 0:
            # Need to set visibility to True to ensure the markers are visible.
            self._inactive_ref_motion_markers.set_visibility(True)
            self._inactive_ref_motion_markers.visualize(inactive_body_pos)
        else:
            self._inactive_ref_motion_markers.set_visibility(False)

        if active_joint_angle_pos.shape[0] != 0:
            self.ref_joint_angle_markers.set_visibility(True)
            self.ref_joint_angle_markers.visualize(active_joint_angle_pos)
        else:
            self.ref_joint_angle_markers.set_visibility(False)

        if root_pos.shape[0] != 0:
            self.root_markers.set_visibility(True)
            self.root_markers.visualize(root_pos, root_rot)
        else:
            self.root_markers.set_visibility(False)
