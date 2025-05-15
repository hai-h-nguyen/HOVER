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

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [".*shoulder.*link.*", ".*elbow.*link.*", ".*hand.*link", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["waist.*joint.*", ".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder.*link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle.*link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
    "vr": {
        "upper_body": [".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["waist.*joint.*", "root.*"],
    }
}

@dataclass
class NeuralWBCEnvCfgRealG1(NeuralWBCEnvCfg):
    decimation = 1
    dt = 0.02  # 50 Hz
    cmd_publish_dt = 0.005  # 200 Hz
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [0, 0]
    default_rfi_lim = 0

    robot = "unitree_g1"

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

    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "torso_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.25, 0., 0.], [0.25, 0., 0.], [0., 0., 0.42]])

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

    control_type: Literal["Pos", "Torque", "None"] = "None"
    robot_actuation_type: Literal["Pos", "Torque"] = "Pos"

    # hardware parameters
    # network_interface = "eno1"
    network_interface = "lo"
    state_channel = "rt/lowstate"
    command_channel = "rt/lowcmd"
    information_channel = "rt/info4debug"
    vis_ref_motion_channel = "rt/refmotion"
    subscriber_freq = 10
    reset_duration = 3.0   # seconds
    reset_step_dt = 0.02  # seconds
    robot_command_mode = "position"  # position or torque
    gravity_value = -9.8  # m/s^2

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
    ]

    # Motor ids
    motor_id_to_name = {
        # Left leg
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
    }

    # control parameters
    stiffness = {
        "left_hip_pitch_joint": 100.,
        "left_hip_roll_joint": 100.,
        "left_hip_yaw_joint": 100.,
        "left_knee_joint": 200.,
        "left_ankle_pitch_joint": 20.,
        "left_ankle_roll_joint": 20.,

        "right_hip_pitch_joint": 100.,
        "right_hip_roll_joint": 100.,
        "right_hip_yaw_joint": 100.,
        "right_knee_joint": 200.,
        "right_ankle_pitch_joint": 20.,
        "right_ankle_roll_joint": 20.,
        
        "waist_yaw_joint": 120.,
        "waist_roll_joint": 120.,
        "waist_pitch_joint": 120.,
        
        "left_shoulder_pitch_joint": 40.,
        "left_shoulder_roll_joint": 40.,
        "left_shoulder_yaw_joint": 40.,
        "left_elbow_joint": 60.,
        
        "right_shoulder_pitch_joint": 40.,
        "right_shoulder_roll_joint": 40.,
        "right_shoulder_yaw_joint": 40.,
        "right_elbow_joint": 60.,

        # fixed wrists
        "left_wrist_roll_joint": 20.,
        "left_wrist_pitch_joint": 20.,
        "left_wrist_yaw_joint": 20.,
        "right_wrist_roll_joint": 20.,
        "right_wrist_pitch_joint": 20.,
        "right_wrist_yaw_joint": 20.,
    }

    damping = {
        "left_hip_pitch_joint": 2.5,
        "left_hip_roll_joint": 2.5,
        "left_hip_yaw_joint": 2.5,
        "left_knee_joint": 5.,
        "left_ankle_pitch_joint": 0.2,
        "left_ankle_roll_joint": 0.1,

        "right_hip_pitch_joint": 2.5,
        "right_hip_roll_joint": 2.5,
        "right_hip_yaw_joint": 2.5,
        "right_knee_joint": 5.,
        "right_ankle_pitch_joint": 0.2,
        "right_ankle_roll_joint": 0.1,
        
        "waist_yaw_joint": 3.,
        "waist_roll_joint": 3.,
        "waist_pitch_joint": 3.,
        
        "left_shoulder_pitch_joint": 1.,
        "left_shoulder_roll_joint": 1.,
        "left_shoulder_yaw_joint": 1.,
        "left_elbow_joint": 1.5,
        
        "right_shoulder_pitch_joint": 1.,
        "right_shoulder_roll_joint": 1.,
        "right_shoulder_yaw_joint": 1.,
        "right_elbow_joint": 1.5,

        # fixed wrists
        "left_wrist_roll_joint": 1.,
        "left_wrist_pitch_joint": 1.,
        "left_wrist_yaw_joint": 1.,
        "right_wrist_roll_joint": 1.,
        "right_wrist_pitch_joint": 1.,
        "right_wrist_yaw_joint": 1.,
    }

    effort_limit = {
        "left_hip_pitch_joint": 88.,
        "left_hip_roll_joint": 88.,
        "left_hip_yaw_joint": 88., 
        "left_knee_joint": 139.,
        "left_ankle_pitch_joint": 50.,
        "left_ankle_roll_joint": 50.,

        "right_hip_pitch_joint": 88.,
        "right_hip_roll_joint": 88.,
        "right_hip_yaw_joint": 88.,
        "right_knee_joint": 139.,
        "right_ankle_pitch_joint": 50.,
        "right_ankle_roll_joint": 50.,
        
        "waist_yaw_joint": 88.,
        "waist_roll_joint": 50.,
        "waist_pitch_joint": 50.,
        
        "left_shoulder_pitch_joint": 25.,
        "left_shoulder_roll_joint": 25.,
        "left_shoulder_yaw_joint": 25.,
        "left_elbow_joint": 25.,
        
        "right_shoulder_pitch_joint": 25.,
        "right_shoulder_roll_joint": 25.,
        "right_shoulder_yaw_joint": 25.,
        "right_elbow_joint": 25.,
    }

    position_limit = {
        "left_hip_pitch_joint": [-2.5307, 2.8798],
        "left_hip_roll_joint": [-0.5236, 2.9671],
        "left_hip_yaw_joint": [-2.7576, 2.7576],
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
        "base_pos": [0., 0., 0.8],
        "base_quat": [1., 0., 0., 0.],
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

    model_urdf_path = get_data_path("robots/g1/g1_29dof_anneal_23dof.urdf")

    def __post_init__(self):
        self.reference_motion_cfg.robot_name = "g1"
        self.reference_motion_cfg.motion_path = get_data_path("motions/g1_test.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/g1_29dof_anneal_23dof_fitmotionONLY.xml")
