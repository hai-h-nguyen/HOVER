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

from isaaclab.utils import configclass


@configclass
class NeuralWBCRewardCfg_H12:
    # Reward and penalty scales
    scales = {
        "reward_track_body_position_extended": 1.0,
        "reward_track_body_position_vr_key_points": 1.6,
        "reward_track_body_position_feet": 2.1,
        "reward_track_body_rotation_extended": 0.5,
        "reward_track_body_angular_velocities_extended": 0.5,
        "reward_track_body_velocities_extended": 0.5,
        "reward_track_joint_positions": 0.75,
        "reward_track_joint_velocities": 0.5,
        "penalize_torques": -0.000001,
        "penalize_action_changes": -0.5,
        "penalize_feet_orientation": -2.0,
        "penalize_feet_heading_alignment": -0.1,
        "penalize_slippage": -1.0,
        "penalize_by_joint_pos_limits": -10.0,
        "penalize_by_joint_velocity_limits": -5.0,
        "penalize_by_torque_limits": -5,
        "penalize_early_termination": -200.0,
    }

    # Sigmas for exponential terms
    body_pos_lower_body_sigma = 0.1
    body_pos_upper_body_sigma = 0.03
    body_pos_vr_key_points_sigma = 0.03
    body_pos_feet_sigma = 0.03
    body_rot_sigma = 1.0
    body_vel_sigma = 1.0
    body_ang_vel_sigma = 1.0
    joint_pos_sigma = 1.0
    joint_vel_sigma = 1.0

    # Weights for weighted sums
    body_pos_lower_body_weight = 0.5
    body_pos_upper_body_weight = 1.0

    # Limits
    torque_limits_scale = 0.85
    # The order here follows the order in cfg.joint_names
    torque_limits = [
        200.0,
        200.0,
        200.0,
        300.0,
        60.0,
        40.0,
        200.0,
        200.0,
        200.0,
        300.0,
        60.0,
        40.0,
        200.0,
        40.0,
        40.0,
        18.0,
        18.0,
        40.0,
        40.0,
        18.0,
        18.0,
    ]
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        (-0.43, 0.43),
        (-3.14, 2.5),
        (-0.43, 3.14),
        (-0.12, 2.19),
        (-0.897334, 0.523598),
        (-0.261799, 0.261799),
        (-0.43, 0.43),
        (-3.14, 2.5),
        (-3.14, 0.43),
        (-0.12, 2.19),
        (-0.897334, 0.523598),
        (-0.261799, 0.261799),
        (-2.35, 2.35),
        (-3.14, 1.57),
        (-0.38, 3.4),
        (-2.66, 3.01),
        (-0.95, 3.18),
        (-3.14, 1.57),
        (-3.4, 0.38),
        (-3.01, 2.66),
        (-0.95, 3.18),
    ]
    joint_vel_limits_scale = 0.85
    joint_vel_limits = [
        23.0,
        23.0,
        23.0,
        14.0,
        9.0,
        9.0,
        23.0,
        23.0,
        23.0,
        14.0,
        9.0,
        9.0,
        23.0,
        9.0,
        9.0,
        20.0,
        20.0,
        9.0,
        9.0,
        20.0,
        20.0,
    ]
    max_contact_force = 500.0
    max_feet_height_limit_before_contact = 0.25
