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
class NeuralWBCRewardCfg_H12_ALMI:
    # Reward and penalty scales
    scales = {
        "penalize_early_termination": -0.0,
        "penalize_lin_vel_z": -2.0,
        "penalize_ang_vel_xy": -0.5,
        "penalize_orientation": -1.0,
        "penalize_base_height": -10.0,
        "penalize_torques": -1e-5,
        "penalize_joint_velocities": -1e-3,
        "penalize_joint_accelerations": -2.5e-7,
        "penalize_action_changes": -0.01,
        # "penalize_collision": 0.0,
        "penalize_by_joint_pos_limits": -0.0,
        "reward_track_lin_vel": 2.0,
        "reward_track_ang_vel": 1.0,
        "penalize_feet_air_time": 0.0,
        "penalize_feet_contact_forces": -0.01,
        "reward_contact": 0.18,
        "penalize_feet_swing_height": -20.0,
        "reward_alive": 0.15,
        "penalize_contact_no_vel": -0.2,
        "penalize_hip_position": -0.0,
        "reward_feet_distance": 1.0,
        "reward_knee_distance": 0.2,
        "penalize_stand_still": -2.0,
        "penalize_ankle_torque": -5e-5,
        "penalize_ankle_action_rate": -0.02,
        "penalize_stance_base_vel": -1.0,
        "penalize_stumble": -0.0,
    }

    # Sigmas for exponential terms
    lin_vel_sigma = 0.25
    ang_vel_sigma = 0.25

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
    max_contact_force = 700.0
    # max_feet_height_limit_before_contact = 0.25

    base_height_target = 0.95
    feet_swing_height_threshold = 0.08
    min_dist = 0.3
    max_dist = 0.6
