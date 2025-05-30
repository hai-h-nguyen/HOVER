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

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .events import NeuralWBCPlayEventCfg_ALMI, NeuralWBCTrainEventCfg_ALMI
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .rewards import NeuralWBCRewardCfg_H12_ALMI
from .terrain import HARD_ROUGH_TERRAINS_CFG, ROUGH_TERRAINS_CFG, flat_terrain

H12_ALMI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=get_data_path("motion_lib/ALMI/h12_almi.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": -0.4,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.8,
            "left_ankle_pitch_joint": -0.4,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": -0.4,
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": 0.8,
            "right_ankle_pitch_joint": -0.4,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.3,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_knee_joint": 300.0,
                "torso_joint": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_knee_joint": 14.0,
                "torso_joint": 23.0,
            },
            stiffness=0,
            damping=0,
            armature=0.01,
            friction=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 40.0,
            },
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
            armature=0.01,
            friction=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_pitch_joint": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_pitch_joint": 20.0,
            },
            stiffness=0,
            damping=0,
            armature=0.01,
            friction=0.01,
        ),
    },
)


@configclass
class NeuralWBCEnvCfgH12_ALMI(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 12
    observation_space = 65
    state_space = 91
    # decimation = 10
    # dt = 0.002

    # Distillation parameters:
    single_history_dim = 60
    observation_history_length = 5

    distill_mask_modes = None
    distill_mask_sparsity_randomization_enabled = None
    tracked_body_names = None

    robot: ArticulationCfg = H12_ALMI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_pitch_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_pitch_link",
    ]

    # Joint names by the order in the MJCF model.
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
        "left_elbow_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20]  # torso, shoulders, elbows

    base_name = "pelvis"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

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
        "left_elbow_pitch_joint": 80.0,
        "right_shoulder_pitch_joint": 120.0,
        "right_shoulder_roll_joint": 120.0,
        "right_shoulder_yaw_joint": 120.0,
        "right_elbow_pitch_joint": 80.0,
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
        "left_elbow_pitch_joint": 1.0,
        "right_shoulder_pitch_joint": 2.0,
        "right_shoulder_roll_joint": 2.0,
        "right_shoulder_yaw_joint": 2.0,
        "right_elbow_pitch_joint": 1.0,
    }

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "torso_link",
    ]

    undesired_contact_body_names = ["pelvis"]
    gravity_x_threshold = 0.8
    gravity_y_threshold = 1.0

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    rewards = NeuralWBCRewardCfg_H12_ALMI()

    policy_obs_noise_scales = {
        "dof_pos": 0.01,
        "dof_vel": 1.5,
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "gravity": 0.05,
        "height_measurements": 0.1,
    }

    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "height_measurements": 5.0,
    }

    motion_path = get_data_path("motions/ALMI/kit_and_custom.pkl")
    mean_episode_length_path = get_data_path("curriculum/ALMI/mean_episode_length_kit_and_custom.csv")

    init_arm_weight = 0.0
    arm_curriculum = True

    init_motion_weight = 0.0
    motion_leading = False

    motion_weight_increase = 0.005
    motion_weight_decrease = 0.01
    arm_weight_increase = 0.005
    arm_weight_decrease = 0.01
    motion_range = 0.0025

    commands_curriculum = True
    max_commands_curriculum = 1.0
    num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    resampling_time = 8.0  # time before command are changed[s]
    heading_command = False  # if true: compute ang vel command from heading error
    commands_range = {
        "lin_vel_x": [-0.7, 0.7],  # m/s
        "lin_vel_y": [-0.3, 0.3],  # m/s
        "ang_vel_yaw": [-0.5, 0.5],  # rad/s
        "heading": [-3.14, 3.14],  # rad
    }

    def __post_init__(self):
        super().__post_init__()

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.ctrl_delay_step_range = (0, 2)
            self.events = NeuralWBCTrainEventCfg_ALMI()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            # self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
        elif self.mode == NeuralWBCModes.TEST:
            self.episode_length_s = 20.0
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg_ALMI()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
