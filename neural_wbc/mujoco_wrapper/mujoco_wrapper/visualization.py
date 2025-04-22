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
from neural_wbc.core import math_utils

from scipy.spatial.transform import Rotation as R

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

        self._viewer.add_marker(
                    pos=np.zeros(3),
                    size=0.05,
                    rgba=(1, 0, 0, 1),
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    label="",
                    id=-1,
                )

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
            root_rot = math_utils.euler_xyz_from_quat(state.root_rot.detach().cpu())
            scale = abs(np.squeeze(state.root_lin_vel.detach().cpu().numpy())[0])
            self._viewer.add_marker(type=mj.mjtGeom.mjGEOM_ARROW,
                      pos=root_pos,
                      label=" ",
                      mat=R.from_euler('xyz', [root_rot[0].item(), root_rot[1].item(), root_rot[2].item()], degrees=False).as_matrix(),
                      size=(0.02, 0.02, 0.15 + scale),
                      rgba=(0, 1, 0, 0.8),
                      emission=1)


    def close(self):
        self._viewer.close()
