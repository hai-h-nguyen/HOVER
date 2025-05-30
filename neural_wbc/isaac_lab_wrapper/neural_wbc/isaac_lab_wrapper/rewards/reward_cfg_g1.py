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
class NeuralWBCRewardCfg_G1:
    # Reward and penalty scales
    scales = {
        "reward_track_body_position_extended": 1.0,
        "reward_track_body_position_vr_key_points": 2.0,
        "reward_track_body_position_feet": 3.0,
        "reward_track_body_rotation_extended": 0.6,  # 0.6
        "reward_track_body_angular_velocities_extended": 0.2,  # 0.2
        "reward_track_body_velocities_extended": 0.2,  # 0.2
        "reward_track_joint_positions": 1.0,
        "reward_track_joint_velocities": 0.5,
        "penalize_torques": -0.000001,
        "penalize_action_changes": -0.5,
        "penalize_feet_orientation": -2.0,
        "penalize_feet_heading_alignment": -0.1,
        "penalize_slippage": -1.0,
        "penalize_by_joint_pos_limits": -10.0,
        "penalize_by_joint_velocity_limits": -5.0,
        "penalize_by_torque_limits": -5.0,
        "penalize_early_termination": -200.0,
    }

    # Sigmas for exponential terms
    body_pos_lower_body_sigma = 0.03  # 0.03
    body_pos_upper_body_sigma = 0.01  # 0.01
    body_pos_vr_key_points_sigma = 0.01  # 0.01
    body_pos_feet_sigma = 0.01  # 0.01
    body_rot_sigma = 0.1  # 0.1
    body_vel_sigma = 10.0  # 10.0
    body_ang_vel_sigma = 10.0  # 10.0
    joint_pos_sigma = 0.1  # 0.1
    joint_vel_sigma = 1.0  # 10.0

    # Weights for weighted sums
    body_pos_lower_body_weight = 1.0  # 0.75
    body_pos_upper_body_weight = 1.0

    # Limits
    torque_limits_scale = 0.85
    # The order here follows the order in cfg.joint_names
    torque_limits = [
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
    ]
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        (-2.5307, 2.8798),
        (-0.5236, 2.9671),
        (-2.7576, 2.7576),
        (-0.087267, 2.8798),
        (-0.87267, 0.5236),
        (-0.2618, 0.2618),
        (-2.5307, 2.8798),
        (-2.9671, 0.5236),
        (-2.7576, 2.7576),
        (-0.087267, 2.8798),
        (-0.87267, 0.5236),
        (-0.2618, 0.2618),
        (-2.618, 2.618),
        (-0.52, 0.52),
        (-0.52, 0.52),
        (-3.0892, 2.6704),
        (-1.5882, 2.2515),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
        (-3.0892, 2.6704),
        (-2.2515, 1.5882),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
    ]
    joint_vel_limits_scale = 0.85
    joint_vel_limits = [
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
    ]
    max_contact_force = 500.0
    max_feet_height_limit_before_contact = 0.25


@configclass
class NeuralWBCRewardCfg_G1_4_DeltaAction:
    # Reward and penalty scales
    scales = {
        "reward_track_body_position_extended": 1.0,
        "reward_track_body_position_vr_key_points": 2.0,
        "reward_track_body_position_feet": 3.0,
        "reward_track_body_rotation_extended": 0.6,  # 0.6
        "reward_track_body_angular_velocities_extended": 0.2,  # 0.2
        "reward_track_body_velocities_extended": 0.2,  # 0.2
        "reward_track_joint_positions": 1.0,
        "reward_track_joint_velocities": 0.5,
        "penalize_action_changes": -0.5,
        "penalize_action_norm": -0.5,
        "penalize_by_joint_pos_limits": -10.0,
        "penalize_by_joint_velocity_limits": -5.0,
        "penalize_by_torque_limits": -5.0,
        "penalize_early_termination": -200.0,
    }

    # Sigmas for exponential terms
    body_pos_lower_body_sigma = 0.03  # 0.03
    body_pos_upper_body_sigma = 0.01  # 0.01
    body_pos_vr_key_points_sigma = 0.01  # 0.01
    body_pos_feet_sigma = 0.01  # 0.01
    body_rot_sigma = 0.1  # 0.1
    body_vel_sigma = 10.0  # 10.0
    body_ang_vel_sigma = 10.0  # 10.0
    joint_pos_sigma = 0.1  # 0.1
    joint_vel_sigma = 1.0  # 10.0

    # Weights for weighted sums
    body_pos_lower_body_weight = 1.0  # 0.75
    body_pos_upper_body_weight = 1.0

    # Limits
    torque_limits_scale = 0.85
    # The order here follows the order in cfg.joint_names
    torque_limits = [
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
    ]
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        (-2.5307, 2.8798),
        (-0.5236, 2.9671),
        (-2.7576, 2.7576),
        (-0.087267, 2.8798),
        (-0.87267, 0.5236),
        (-0.2618, 0.2618),
        (-2.5307, 2.8798),
        (-2.9671, 0.5236),
        (-2.7576, 2.7576),
        (-0.087267, 2.8798),
        (-0.87267, 0.5236),
        (-0.2618, 0.2618),
        (-2.618, 2.618),
        (-0.52, 0.52),
        (-0.52, 0.52),
        (-3.0892, 2.6704),
        (-1.5882, 2.2515),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
        (-3.0892, 2.6704),
        (-2.2515, 1.5882),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
    ]
    joint_vel_limits_scale = 0.85
    joint_vel_limits = [
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
    ]
    max_contact_force = 500.0
    max_feet_height_limit_before_contact = 0.25
