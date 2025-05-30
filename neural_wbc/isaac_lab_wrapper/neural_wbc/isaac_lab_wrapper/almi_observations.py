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
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .neural_wbc_env import NeuralWBCEnv

from neural_wbc.core import math_utils
from neural_wbc.core.body_state import BodyState
from neural_wbc.core.observations import compute_almi_observations


def compute_observations(
    env: NeuralWBCEnv,
    body_state: BodyState,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene.articulations[asset_cfg.name]

    deafult_joint_pos = asset.data.default_joint_pos[:, env._joint_ids]

    obs_dict = {}
    # First collect teacher policy observations.
    base_id = env._body_names.index(env.base_name)
    almi_obs = compute_almi_observations(
        base_id=base_id,
        body_state=body_state,
        projected_gravity=asset.data.projected_gravity_b,
        commands=env.commands[:, :3] * env.commands_scales,
        dof_pos=(body_state.joint_pos - deafult_joint_pos) * env.cfg.obs_scales["dof_pos"],
        dof_vel=body_state.joint_vel * env.cfg.obs_scales["dof_vel"],
        last_actions=env.actions,
        left_sin_phase=env.left_sin_phase,
        right_sin_phase=env.right_sin_phase,
        # history=env.history.entries,
        obs_scales=env.cfg.obs_scales,
    )
    obs_dict["teacher_policy"] = almi_obs

    local_base_lin_vel = math_utils.quat_rotate_inverse(
        body_state.body_rot_extend[:, base_id, :], body_state.body_lin_vel_extend[:, base_id, :]
    )

    # Then the privileged observations.
    privileged_obs, privileged_obs_dict = compute_privileged_observations(
        env=env, asset=asset, local_base_lin_vel=local_base_lin_vel
    )
    obs_dict.update(privileged_obs_dict)
    obs_dict["critic"] = torch.cat([almi_obs, privileged_obs], dim=1)
    return obs_dict


def compute_privileged_observations(
    env: NeuralWBCEnv, asset: Articulation, local_base_lin_vel: torch.Tensor | None = None
):
    contact_forces = env.contact_sensor.data.net_forces_w[:, env.feet_ids, :]

    privileged_obs_dict = {
        "base_lin_vel": local_base_lin_vel,
        # "base_com_bias": env.base_com_bias.to(env.device),
        "ground_friction_values": asset.data.joint_friction[:, env.feet_ids],
        "body_mass_scale": env.body_mass_scale,
        # "kp_scale": env.kp_scale,
        # "kd_scale": env.kd_scale,
        # "rfi_lim_scale": env.rfi_lim / env.cfg.default_rfi_lim,
        "contact_forces": contact_forces.reshape(contact_forces.shape[0], -1),
        "recovery_counters": torch.clamp_max(env.recovery_counters.unsqueeze(1), 1),
        # "joint_friction": asset.data.joint_friction,
        # "joint_armature": asset.data.joint_armature,
    }
    privileged_obs = torch.cat(
        [tensor for tensor in privileged_obs_dict.values()],
        dim=-1,
    )
    return privileged_obs, privileged_obs_dict
