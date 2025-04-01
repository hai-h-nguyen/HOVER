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


import numpy as np

import mujoco as mj
import mujoco_viewer.mujoco_viewer as mjv

from neural_wbc.core.reference_motion import ReferenceMotionState

from scipy.spatial.transform import Rotation

def quaternion_to_rotation_matrix(wxyz):
    w, x, y, z = wxyz
    x, z = z, y
    
    # Calculate the squares of each component
    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    # Compute the rotation matrix components
    R11 = 1 - 2 * (y2 + z2)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)
    
    R21 = 2 * (x * y + w * z)
    R22 = 1 - 2 * (x2 + z2)
    R23 = 2 * (y * z - w * x)
    
    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = 1 - 2 * (x2 + y2)
    
    # Return the rotation matrix
    rotation_matrix = np.array([[R11, R12, R13],
                                [R21, R22, R23],
                                [R31, R32, R33]])
    
    return rotation_matrix

class MujocoVisualizer:
    """A basic Mujoco visualizer"""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: int = 1400,
        height: int = 1200,
        start_paused: bool = True,
    ):
        self._model = model
        self._data = data
        self._viewer = mjv.MujocoViewer(
            self._model,
            self._data,
            width=width,
            height=height,
            hide_menus=True,
        )

        self._viewer._paused = start_paused  # pylint: disable=W0212

    def update(self):
        """Update the Mujoco viewer."""
        if self._viewer.is_alive:
            self._viewer.render()

    def draw_reference_state(self, mask, state: ReferenceMotionState):
        """Visualize the reference state in Mujoco."""
        mask = np.squeeze(mask.detach().cpu().numpy())
        body_pos_extend_np = np.squeeze(state.body_pos_extend.detach().cpu().numpy())
        num_bodies = body_pos_extend_np.shape[0]

        for i in range(num_bodies):
            if mask[i]:
                self._viewer.add_marker(
                    pos=body_pos_extend_np[i],
                    size=0.05,
                    rgba=(1, 0, 0, 1),
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    label="",
                    id=i,
                )
        
        if mask[-7:].sum() > 0:
            root_pos = np.squeeze(state.root_pos.detach().cpu().numpy())
            root_pos[2] += 1.0
            root_rot = np.squeeze(state.root_rot.detach().cpu().numpy())
            root_vel = np.squeeze(state.root_lin_vel.detach().cpu().numpy())
            scale = np.linalg.norm(root_vel)
            self._viewer.add_marker(type=mj.mjtGeom.mjGEOM_ARROW,
                      pos=root_pos,
                      label=" ",
                      mat=quaternion_to_rotation_matrix(root_rot),
                      size=(0.02, 0.02, 0.15 + scale),
                      rgba=(0, 1, 0, 0.8),
                      emission=1)


    def close(self):
        self._viewer.close()
