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
from __future__ import annotations

import torch

from neural_wbc.core import math_utils

from ..body_state import BodyState


def compute_almi_observations(
    base_id: int,
    body_state: BodyState,
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    last_actions: torch.Tensor,
    left_sin_phase: torch.Tensor,
    right_sin_phase: torch.Tensor,
    # history: torch.Tensor,
    local_base_ang_velocity: torch.Tensor | None = None,
    obs_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes observations for a almi policy."""

    local_base_ang_vel = local_base_ang_velocity
    if local_base_ang_velocity is None:
        local_base_ang_vel = math_utils.quat_rotate_inverse(
            body_state.body_rot_extend[:, base_id, :], body_state.body_ang_vel_extend[:, base_id, :]
        )
    local_base_ang_vel *= obs_scales["ang_vel"]
    obs = torch.cat(
        [
            local_base_ang_vel,
            projected_gravity,
            commands,
            dof_pos,
            dof_vel,
            last_actions,
            left_sin_phase,
            right_sin_phase,
        ],
        dim=-1,
    )

    return obs
