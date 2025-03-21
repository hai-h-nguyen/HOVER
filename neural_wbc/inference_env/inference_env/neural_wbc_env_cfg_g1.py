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
class NeuralWBCEnvCfgG1(NeuralWBCEnvCfg):
    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [2, 2]
    default_rfi_lim = 0
    robot = "mujoco_robot"

    body_names = [
                  'pelvis',
                  'left_hip_pitch_link',
                  'left_hip_roll_link',
                  'left_hip_yaw_link',
                  'left_knee_link',
                  'left_ankle_pitch_link',
                  'left_ankle_roll_link',
                  'right_hip_pitch_link',
                  'right_hip_roll_link',
                  'right_hip_yaw_link',
                  'right_knee_link',
                  'right_ankle_pitch_link',
                  'right_ankle_roll_link',
                  'waist_yaw_link',
                  'waist_roll_link',
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
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
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
    extend_body_pos = torch.tensor([[0.25, 0, 0], [0.25, 0, 0], [0, 0, 0.42]])

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
    single_history_dim = 75
    observation_history_length = 25
    num_bodies = 24
    num_joints = 23
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    control_type = "Pos"
    robot_actuation_type = "Torque"  # Pos or Torque

    # control parameters
    stiffness = {
        "left_hip_pitch_joint": 100.0,
        "left_hip_roll_joint": 100.0,
        "left_hip_yaw_joint": 100.0,
        "left_knee_joint": 200.0,
        "left_ankle_pitch_joint": 20.0,
        "left_ankle_roll_joint": 20.0,

        "right_hip_pitch_joint": 100.0,
        "right_hip_roll_joint": 100.0,
        "right_hip_yaw_joint": 100.0,
        "right_knee_joint": 200.0,
        "right_ankle_pitch_joint": 20.0,
        "right_ankle_roll_joint": 20.0,
        
        "waist_yaw_joint": 400,
        "waist_roll_joint": 400,
        "waist_pitch_joint": 400,
        
        "left_shoulder_pitch_joint": 90.0,
        "left_shoulder_roll_joint": 60.0,
        "left_shoulder_yaw_joint": 20.0,
        "left_elbow_joint": 60.0,
        
        "right_shoulder_pitch_joint": 90.0,
        "right_shoulder_roll_joint": 60.0,
        "right_shoulder_yaw_joint": 20.0,
        "right_elbow_joint": 60.0,
    }

    damping = {
        "left_hip_pitch_joint": 2.5,
        "left_hip_roll_joint": 2.5,
        "left_hip_yaw_joint": 2.5,
        "left_knee_joint": 5.0,
        "left_ankle_pitch_joint": 0.2,
        "left_ankle_roll_joint": 0.1,

        "right_hip_pitch_joint": 2.5,
        "right_hip_roll_joint": 2.5,
        "right_hip_yaw_joint": 2.5,
        "right_knee_joint": 5.0,
        "right_ankle_pitch_joint": 0.2,
        "right_ankle_roll_joint": 0.1,
        
        "waist_yaw_joint": 5.0,
        "waist_roll_joint": 5.0,
        "waist_pitch_joint": 5.0,
        
        "left_shoulder_pitch_joint": 2.0,
        "left_shoulder_roll_joint": 1.0,
        "left_shoulder_yaw_joint": 0.4,
        "left_elbow_joint": 1.0,
        
        "right_shoulder_pitch_joint": 2.0,
        "right_shoulder_roll_joint": 1.0,
        "right_shoulder_yaw_joint": 0.4,
        "right_elbow_joint": 1.0,
    }

    effort_limit = {
        "left_hip_pitch_joint": 88.0,
        "left_hip_roll_joint": 88.0,
        "left_hip_yaw_joint": 88.0, 
        "left_knee_joint": 139.0,
        "left_ankle_pitch_joint": 50.0,
        "left_ankle_roll_joint": 50.0,

        "right_hip_pitch_joint": 88.0,
        "right_hip_roll_joint": 88.0,
        "right_hip_yaw_joint": 88.0,
        "right_knee_joint": 139.0,
        "right_ankle_pitch_joint": 50.0,
        "right_ankle_roll_joint": 50.0,
        
        "waist_yaw_joint": 88.0,
        "waist_roll_joint": 50.0,
        "waist_pitch_joint": 50.0,
        
        "left_shoulder_pitch_joint": 25.0,
        "left_shoulder_roll_joint": 25.0,
        "left_shoulder_yaw_joint": 25.0,
        "left_elbow_joint": 25.0,
        
        "right_shoulder_pitch_joint": 25.0,
        "right_shoulder_roll_joint": 25.0,
        "right_shoulder_yaw_joint": 25.0,
        "right_elbow_joint": 25.0,
    }

    position_limit = {
        "left_hip_pitch_joint": [-2.5307, 2.8798],
        "left_hip_roll_joint": [-0.5236, 2.9671],
        "left_hip_yaw_joint": [-2,7576, 2.7576],
        "left_knee_joint": [-0.087267, 2.8798],
        "left_ankle_pitch_joint": [-0.87267, 0.5236],
        "left_ankle_roll_joint": [-0.2618, 0.2618],

        "right_hip_pitch_joint": [-2.5307, 2.8798],
        "right_hip_roll_joint": [-2.9671, 0.5236],
        "right_hip_yaw_joint": [-2.7576, 2.7576],
        "right_knee_joint": [-0.087267, 2.8798],
        "right_ankle_pitch_joint": [-0.87267, 0.5236],
        "right_ankle_roll_joint": [-0.2618, 0.2618],
        
        "waist_yaw_joint": [-2.618, 2.618],
        "waist_roll_joint": [-0.52, 0.52],
        "waist_pitch_joint": [-0.52, 0.52],
        
        "left_shoulder_pitch_joint": [-3.0892, 2.6704],
        "left_shoulder_roll_joint": [-1.5882, 2.2515],
        "left_shoulder_yaw_joint": [-2.618, 2.618],
        "left_elbow_joint": [-1.0472, 2.0944],
        
        "right_shoulder_pitch_joint": [-3.0892, 2.6704],
        "right_shoulder_roll_joint": [-2.2515, 1.5882],
        "right_shoulder_yaw_joint": [-2.618, 2.618],
        "right_elbow_joint": [-1.0472, 2.0944],
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 0.8],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
                    "left_hip_pitch_joint": -0.1,
                    "left_hip_roll_joint": 0.,
                    "left_hip_yaw_joint": 0.,
                    "left_knee_joint": 0.3,
                    "left_ankle_pitch_joint": -0.2,
                    "left_ankle_roll_joint": 0.,

                    "right_hip_pitch_joint": -0.1,
                    "right_hip_roll_joint": 0.,
                    "right_hip_yaw_joint": 0.,
                    "right_knee_joint": 0.3,
                    "right_ankle_pitch_joint": -0.2,
                    "right_ankle_roll_joint": 0.,
                    
                    "waist_yaw_joint" : 0.,
                    "waist_roll_joint" : 0.,
                    "waist_pitch_joint" : 0.,
                    
                    "left_shoulder_pitch_joint": 0.,
                    "left_shoulder_roll_joint": 0.,
                    "left_shoulder_yaw_joint": 0.,
                    "left_elbow_joint": 0.,
                    
                    "right_shoulder_pitch_joint": 0.,
                    "right_shoulder_roll_joint": 0.,
                    "right_shoulder_yaw_joint": 0.,
                    "right_elbow_joint": 0.,
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # hips, knees, ankles, torso
    upper_body_joint_ids = [15, 16, 17, 18, 19, 20, 21, 22]  # torso, shoulders, elbows

    def __post_init__(self):
        self.reference_motion_cfg.robot_name = "g1"
        self.reference_motion_cfg.motion_path = get_data_path("motions/amass_full_g1_anneal.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/g1_29dof_anneal_23dof_fitmotionONLY.xml")
