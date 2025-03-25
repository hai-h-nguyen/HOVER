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
from dataclasses import dataclass

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgH12(NeuralWBCEnvCfg):
    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [2, 2]
    default_rfi_lim = 0
    robot = "mujoco_robot"
    
    body_names = [
                  'pelvis',
                  
                  'left_hip_yaw_link',
                  'left_hip_pitch_link',
                  'left_hip_roll_link',
                  'left_knee_link',
                  'left_ankle_pitch_link',
                  'left_ankle_roll_link',
                  
                  'right_hip_yaw_link',
                  'right_hip_pitch_link',
                  'right_hip_roll_link',
                  'right_knee_link',
                  'right_ankle_pitch_link',
                  'right_ankle_roll_link',

                  'torso_link',
                  
                  'left_shoulder_pitch_link',
                  'left_shoulder_roll_link',
                  'left_shoulder_yaw_link',
                  'left_elbow_link',

                  'right_shoulder_pitch_link',
                  'right_shoulder_roll_link',
                  'right_shoulder_yaw_link',
                  'right_elbow_link',
                ]
    joint_names = [
                    "left_hip_yaw_joint",
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",

                    "right_hip_yaw_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",

                    "torso_joint",

                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",

                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    ]

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "torso_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.7]])

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
    single_history_dim = 69
    observation_history_length = 25
    num_bodies = 22
    num_joints = 21
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    control_type = "Pos"
    robot_actuation_type = "Torque"  # Pos or Torque

    # control parameters
    stiffness = {
        "left_hip_yaw_joint": 200.0,
        "left_hip_pitch_joint": 200.0,
        "left_hip_roll_joint": 200.0,
        "left_knee_joint": 300.0,
        "left_ankle_pitch_joint": 40.0,
        "left_ankle_roll_joint": 40.0,

        "right_hip_yaw_joint": 200.0,
        "right_hip_pitch_joint": 200.0,
        "right_hip_roll_joint": 200.0,
        "right_knee_joint": 300.0,
        "right_ankle_pitch_joint": 40.0,
        "right_ankle_roll_joint": 40.0,

        "torso_joint": 300,

        "left_shoulder_pitch_joint": 120.0,
        "left_shoulder_roll_joint": 120.0,
        "left_shoulder_yaw_joint": 120.0,
        "left_elbow_joint": 80.0,

        "right_shoulder_pitch_joint": 120.0,
        "right_shoulder_roll_joint": 120.0,
        "right_shoulder_yaw_joint": 120.0,
        "right_elbow_joint": 80.0,
    }

    damping = {
        "left_hip_yaw_joint": 2.5,
        "left_hip_pitch_joint": 2.5,
        "left_hip_roll_joint": 2.5,
        "left_knee_joint": 4.0,
        "left_ankle_pitch_joint": 2.0,
        "left_ankle_roll_joint": 2.0,

        "right_hip_yaw_joint": 2.5,
        "right_hip_pitch_joint": 2.5,
        "right_hip_roll_joint": 2.5,
        "right_knee_joint": 4.0,
        "right_ankle_pitch_joint": 2.0,
        "right_ankle_roll_joint": 2.0,

        "torso_joint": 3.0,

        "left_shoulder_pitch_joint": 2.0,
        "left_shoulder_roll_joint": 2.0,
        "left_shoulder_yaw_joint": 2.0,
        "left_elbow_joint": 1.0,

        "right_shoulder_pitch_joint": 2.0,
        "right_shoulder_roll_joint": 2.0,
        "right_shoulder_yaw_joint": 2.0,
        "right_elbow_joint": 1.0,
    }

    effort_limit = {
        "left_hip_yaw_joint": 200.0,
        "left_hip_pitch_joint": 200.0,
        "left_hip_roll_joint": 200.0,
        "left_knee_joint": 300.0,
        "left_ankle_pitch_joint": 60.0,
        "left_ankle_roll_joint": 40.0,

        "right_hip_yaw_joint": 200.0,
        "right_hip_pitch_joint": 200.0,
        "right_hip_roll_joint": 200.0,
        "right_knee_joint": 300.0,
        "right_ankle_pitch_joint": 60.0,
        "right_ankle_roll_joint": 40.0,

        "torso_joint": 200.0,

        "left_shoulder_pitch_joint": 40.0,
        "left_shoulder_roll_joint": 40.0,
        "left_shoulder_yaw_joint": 18.0,
        "left_elbow_joint": 18.0,

        "right_shoulder_pitch_joint": 40.0,
        "right_shoulder_roll_joint": 40.0,
        "right_shoulder_yaw_joint": 18.0,
        "right_elbow_joint": 18.0,
    }

    position_limit = {
        "left_hip_yaw_joint": [-0.43, 0.43],
        "left_hip_pitch_joint": [-3.14, 2.5],
        "left_hip_roll_joint": [-0.43, 3.14],
        "left_knee_joint": [-0.12, 2.19],
        "left_ankle_pitch_joint": [-0.897334, 0.523598],
        "left_ankle_roll_joint": [-0.261799, 0.261799],

        "right_hip_yaw_joint": [-0.43, 0.43],
        "right_hip_pitch_joint": [-3.14, 2.5],
        "right_hip_roll_joint": [-3.14, 0.43],
        "right_knee_joint": [-0.12, 2.19],
        "right_ankle_pitch_joint": [-0.897334, 0.523598],
        "right_ankle_roll_joint": [-0.261977, 0.261799],
        
        "torso_joint": [-2.35, 2.35],
        
        "left_shoulder_pitch_joint": [-3.14, 1.57],
        "left_shoulder_roll_joint": [-0.38, 3.4],
        "left_shoulder_yaw_joint": [-2.66, 3.01],
        "left_elbow_joint": [-0.95, 3.18],
        
        "right_shoulder_pitch_joint": [-3.14, 1.57],
        "right_shoulder_roll_joint": [-3.4, 0.38],
        "right_shoulder_yaw_joint": [-3.01, 2.66],
        "right_elbow_joint": [-0.95, 3.18],
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 1.05],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.28,
            "left_knee_joint": 0.79,
            "left_ankle_pitch_joint": -0.52,
            "left_ankle_roll_joint": -0.52,

            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.28,
            "right_knee_joint": 0.79,
            "right_ankle_pitch_joint": -0.52,
            "right_ankle_roll_joint": -0.52,

            "torso_joint": 0.0,
            
            "left_shoulder_pitch_joint": 0.28,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.52,

            "right_shoulder_pitch_joint": 0.28,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.52,
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20]  # torso, shoulders, elbows

    def __post_init__(self):
        self.reference_motion_cfg.robot_name = "h12"
        self.reference_motion_cfg.motion_path = get_data_path("motions/h12_test.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/h1_2_fitmotionONLY.xml")
