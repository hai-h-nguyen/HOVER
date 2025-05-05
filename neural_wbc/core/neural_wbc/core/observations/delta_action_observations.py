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

import phc.utils.torch_utils as torch_utils

from neural_wbc.core import math_utils

from .student_observations import compute_distilled_robot_state_observation, compute_distilled_imitation_observations
from ..body_state import BodyState
from ..reference_motion import ReferenceMotionState


def compute_delta_action_observations(
    base_id: int,
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    projected_gravity: torch.Tensor,
    last_actions: torch.Tensor,
    history: torch.Tensor,
    mask: torch.Tensor,
    ref_episodic_offset: torch.Tensor | None = None,
    local_base_ang_velocity: torch.Tensor | None = None,
    recorded_actions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes observations for a student policy."""
    obs_dict = {
        "distilled_robot_state": compute_distilled_robot_state_observation(
            body_state=body_state,
            base_id=base_id,
            projected_gravity=projected_gravity,
            local_base_ang_velocity=local_base_ang_velocity,
        ),
        "distilled_imitation": compute_distilled_imitation_observations(
            ref_motion_state=ref_motion_state,
            body_state=body_state,
            mask=mask,
            ref_episodic_offset=ref_episodic_offset,
        ),
        "distilled_last_action": last_actions,
        "distilled_historical_info": history,
        "recorded_actions": recorded_actions,
    }
    
    obs = torch.cat(
        [tensor for tensor in obs_dict.values()],
        dim=-1,
    )

    return obs, obs_dict

