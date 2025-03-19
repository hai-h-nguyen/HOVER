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
from typing import Literal

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgRealG1(NeuralWBCEnvCfg):
    decimation = 1
    dt = 0.02  # 50 Hz
    cmd_publish_dt = 0.005  # 200 Hz
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [0, 0]
    default_rfi_lim = 0
<<<<<<< HEAD
    robot = "unitree_g1"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "torso_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.25, 0, 0], [0.25, 0, 0], [0, 0, 0.42]])
=======
    robot = "unitree_h1"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "pelvis"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])
>>>>>>> d16ac8a (fixing deployment)

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
<<<<<<< HEAD
    single_history_dim = 75
    observation_history_length = 25
    num_bodies = 24
    num_joints = 23
=======
    single_history_dim = 63
    observation_history_length = 25
    num_bodies = 20
    num_joints = 19
>>>>>>> d16ac8a (fixing deployment)
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    control_type: Literal["Pos", "Torque", "None"] = "None"
    robot_actuation_type: Literal["Pos", "Torque"] = "Pos"

    # hardware parameters
<<<<<<< HEAD
    # network_interface = "eno1"
    network_interface = "lo"
    state_channel = "rt/lowstate"
    command_channel = "rt/lowcmd"
    subscriber_freq = 10
    reset_duration = 2.0  # seconds
=======
    network_interface = "enx0c3796b54c40"
    state_channel = "rt/lowstate"
    command_channel = "rt/lowcmd"
    subscriber_freq = 10
    reset_duration = 10.0  # seconds
>>>>>>> d16ac8a (fixing deployment)
    reset_step_dt = 0.01  # seconds
    robot_command_mode = "position"  # position or torque
    gravity_value = -9.8  # m/s^2

<<<<<<< HEAD
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
    
    # Stiffness and damping parameters
    JointSeq2MotorID = [
        0,
        1,
        2,
        3,
        4,  
        5,# left leg
        6,
        7,
        8,
        9,  
        10,  
        11,# right leg
        12,
        13,
        14,# waist
        15,
        16,
        17,
        18,# left arm
        22,
        23,
        24,
        25,# left arm
=======
    # Command packets
    head0 = 0xFE
    head1 = 0xEF
    level_flag = 0xFF
    gpio = 0
    weak_motor_mode = 0x01
    strong_motor_mode = 0x0A

    # Stiffness and damping parameters
    kp_low = 60.0
    kp_high = 200.0
    kd_low = 1.5
    kd_high = 5.0

    JointSeq2MotorID = [
        7,
        3,
        4,
        5,
        10,  # left leg
        8,
        0,
        1,
        2,
        11,  # right leg
        6,  # torso
        16,
        17,
        18,
        19,  # left arm
        12,
        13,
        14,
        15,  # right arm
    ]

    MotorID2JointSeq = [
        6,  # right hip roll
        7,  # right hip pitch
        8,  # right knee
        1,  # left hip roll
        2,  # left hip pitch
        3,  # left knee
        10,  # torso
        0,  # left hip yaw
        5,  # right hip yaw
        99999,  # N/A for motor index 9
        4,  # left ankle
        9,  # right ankle
        15,  # right shoulder pitch
        16,  # right shoulder roll
        17,  # right should yaw
        18,  # right elbow
        11,  # left shoulder pitch
        12,  # left shoulder roll
        13,  # left shoulder yaw
        14,  # left elbow
>>>>>>> d16ac8a (fixing deployment)
    ]

    # Motor ids
    motor_id_to_name = {
        # Left leg
<<<<<<< HEAD
        0: "left_hip_pitch",
        1: "left_hip_roll",
        2: "left_hip_yaw",
        3: "left_knee",
        4: "left_ankle_pitch",
        5: "left_ankle_roll",
        6: "right_hip_pitch",
        7: "right_hip_roll",
        8: "right_hip_yaw",
        9: "right_knee",
        10: "right_ankle_pitch",
        11: "right_ankle_roll",
        12: "waist_yaw",
        13: "waist_roll",
        14: "waist_pitch",
        15: "left_shoulder_pitch",
        16: "left_shoulder_roll",
        17: "left_shoulder_yaw",
        18: "left_elbow",
        22: "right_shoulder_pitch",
        23: "right_shoulder_roll",
        24: "right_shoulder_yaw",
        25: "right_elbow",
    }

    # fixed wrists
    wrist_motor_id_to_name = {
        19: "left_wrist_roll",
        20: "left_wrist_pitch",
        21: "left_wrist_yaw",
        26: "right_wrist_roll",
        27: "right_wrist_pitch",
        28: "right_wrist_yaw",
=======
        7: "left_hip_yaw",
        3: "left_hip_roll",
        4: "left_hip_pitch",
        5: "left_knee",
        10: "left_ankle",
        # Right leg
        8: "right_hip_yaw",
        0: "right_hip_roll",
        1: "right_hip_pitch",
        2: "right_knee",
        11: "right_ankle",
        # Torso
        6: "torso",
        # Left arm
        16: "left_shoulder_pitch",
        17: "left_shoulder_roll",
        18: "left_shoulder_yaw",
        19: "left_elbow",
        # Right arm
        12: "right_shoulder_pitch",
        13: "right_shoulder_roll",
        14: "right_shoulder_yaw",
        15: "right_elbow",
        # unused id
        9: "unused_motor",
    }

    weak_motors = {
        "left_ankle",
        "right_ankle",
        # Left arm
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        # Right arm
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
>>>>>>> d16ac8a (fixing deployment)
    }

    # control parameters
    stiffness = {
<<<<<<< HEAD
        "left_hip_pitch_joint": 100.0,
        "left_hip_roll_joint": 100.0,
        "left_hip_yaw_joint": 100.0,
        "left_knee_joint": 150.0,
        "left_ankle_pitch_joint": 40.0,
        "left_ankle_roll_joint": 40.0,

        "right_hip_pitch_joint": 100.0,
        "right_hip_roll_joint": 100.0,
        "right_hip_yaw_joint": 100.0,
        "right_knee_joint": 150.0,
        "right_ankle_pitch_joint": 40.0,
        "right_ankle_roll_joint": 40.0,
        
        "waist_yaw_joint": 300.,
        "waist_roll_joint": 300.,
        "waist_pitch_joint": 300.,
        
        "left_shoulder_pitch_joint": 100.0,
        "left_shoulder_roll_joint": 100.0,
        "left_shoulder_yaw_joint": 50.0,
        "left_elbow_joint": 50.0,
        
        "right_shoulder_pitch_joint": 100.0,
        "right_shoulder_roll_joint": 100.0,
        "right_shoulder_yaw_joint": 50.0,
        "right_elbow_joint": 50.0,

        # fixed wrists
        "left_wrist_roll_joint": 20.0,
        "left_wrist_pitch_joint": 20.0,
        "left_wrist_yaw_joint": 20.0,
        "right_wrist_roll_joint": 20.0,
        "right_wrist_pitch_joint": 20.0,
        "right_wrist_yaw_joint": 20.0,
    }

    damping = {
        "left_hip_pitch_joint": 2.,
        "left_hip_roll_joint": 2.,
        "left_hip_yaw_joint": 2.,
        "left_knee_joint": 4.0,
        "left_ankle_pitch_joint": 2.0,
        "left_ankle_roll_joint": 2.0,

        "right_hip_pitch_joint": 2.,
        "right_hip_roll_joint": 2.,
        "right_hip_yaw_joint": 2.,
        "right_knee_joint": 4.0,
        "right_ankle_pitch_joint": 2.,
        "right_ankle_roll_joint": 2.,
        
        "waist_yaw_joint": 3.0,
        "waist_roll_joint": 3.0,
        "waist_pitch_joint": 3.0,
        
        "left_shoulder_pitch_joint": 2.0,
        "left_shoulder_roll_joint": 2.0,
        "left_shoulder_yaw_joint": 2.0,
        "left_elbow_joint": 2.0,
        
        "right_shoulder_pitch_joint": 2.0,
        "right_shoulder_roll_joint": 2.0,
        "right_shoulder_yaw_joint": 2.0,
        "right_elbow_joint": 2.0,

        # fixed wrists
        "left_wrist_roll_joint": 1.0,
        "left_wrist_pitch_joint": 1.0,
        "left_wrist_yaw_joint": 1.0,
        "right_wrist_roll_joint": 1.0,
        "right_wrist_pitch_joint": 1.0,
        "right_wrist_yaw_joint": 1.0,
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
        "base_pos": [0.0, 0.0, 0.78],
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
=======
        "left_hip_yaw": 150.0,
        "left_hip_roll": 150.0,
        "left_hip_pitch": 200.0,
        "left_knee": 200.0,
        "left_ankle": 20.0,
        "right_hip_yaw": 150.0,
        "right_hip_roll": 150.0,
        "right_hip_pitch": 200.0,
        "right_knee": 200.0,
        "right_ankle": 20.0,
        "torso": 200.0,
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 40.0,
        "left_elbow": 40.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 40.0,
        "right_elbow": 40.0,
    }

    damping = {
        "left_hip_yaw": 5.0,
        "left_hip_roll": 5.0,
        "left_hip_pitch": 5.0,
        "left_knee": 5.0,
        "left_ankle": 4.0,
        "right_hip_yaw": 5.0,
        "right_hip_roll": 5.0,
        "right_hip_pitch": 5.0,
        "right_knee": 5.0,
        "right_ankle": 4.0,
        "torso": 5.0,
        "left_shoulder_pitch": 10.0,
        "left_shoulder_roll": 10.0,
        "left_shoulder_yaw": 10.0,
        "left_elbow": 10.0,
        "right_shoulder_pitch": 10.0,
        "right_shoulder_roll": 10.0,
        "right_shoulder_yaw": 10.0,
        "right_elbow": 10.0,
    }

    effort_limit = {
        "left_hip_yaw": 200.0,
        "left_hip_roll": 200.0,
        "left_hip_pitch": 200.0,
        "left_knee": 300.0,
        "left_ankle": 40.0,
        "right_hip_yaw": 200.0,
        "right_hip_roll": 200.0,
        "right_hip_pitch": 200.0,
        "right_knee": 300.0,
        "right_ankle": 40.0,
        "torso": 200.0,
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 18.0,
        "left_elbow": 18.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 18.0,
        "right_elbow": 18.0,
    }

    position_limit = {
        "left_hip_yaw": [-0.43, 0.43],
        "left_hip_roll": [-0.43, 0.43],
        "left_hip_pitch": [-1.57, 1.57],
        "left_knee": [-0.26, 2.05],
        "left_ankle": [-0.87, 0.52],
        "right_hip_yaw": [-0.43, 0.43],
        "right_hip_roll": [-0.43, 0.43],
        "right_hip_pitch": [-1.57, 1.57],
        "right_knee": [-0.26, 2.05],
        "right_ankle": [-0.87, 0.52],
        "torso": [-2.35, 2.35],
        "left_shoulder_pitch": [-2.87, 2.87],
        "left_shoulder_roll": [-0.34, 3.11],
        "left_shoulder_yaw": [-1.3, 4.45],
        "left_elbow": [-1.25, 2.61],
        "right_shoulder_pitch": [-2.87, 2.87],
        "right_shoulder_roll": [-3.11, 0.34],
        "right_shoulder_yaw": [-4.45, 1.3],
        "right_elbow": [-1.25, 2.61],
    }

    velocity_limit = {
        "left_hip_yaw": 23.0,
        "left_hip_roll": 23.0,
        "left_hip_pitch": 23.0,
        "left_knee": 14.0,
        "left_ankle": 9.0,
        "right_hip_yaw": 23.0,
        "right_hip_roll": 23.0,
        "right_hip_pitch": 23.0,
        "right_knee": 14.0,
        "right_ankle": 9.0,
        "torso": 23.0,
        "left_shoulder_pitch": 9.0,
        "left_shoulder_roll": 9.0,
        "left_shoulder_yaw": 20.0,
        "left_elbow": 20.0,
        "right_shoulder_pitch": 9.0,
        "right_shoulder_roll": 9.0,
        "right_shoulder_yaw": 20.0,
        "right_elbow": 20.0,
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 0.98],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.28,
            "left_knee": 0.79,
            "left_ankle": -0.52,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.28,
            "right_knee": 0.79,
            "right_ankle": -0.52,
            "torso": 0.0,
            "left_shoulder_pitch": 0.28,
            "left_shoulder_roll": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.52,
            "right_shoulder_pitch": 0.28,
            "right_shoulder_roll": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.52,
>>>>>>> d16ac8a (fixing deployment)
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model.
<<<<<<< HEAD
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # hips, knees, ankles, torso
    upper_body_joint_ids = [15, 16, 17, 18, 19, 20, 21, 22]  # torso, shoulders, elbows
=======
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # hips, knees, ankles
    upper_body_joint_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18]  # torso, shoulders, elbows
>>>>>>> d16ac8a (fixing deployment)

    def __post_init__(self):
        self.reference_motion_cfg.robot_name = "g1"
        self.reference_motion_cfg.motion_path = get_data_path("motions/stable_punch.pkl")
<<<<<<< HEAD
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/g1_29dof_anneal_23dof_fitmotionONLY.xml")
=======
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/g1.xml")
>>>>>>> d16ac8a (fixing deployment)
