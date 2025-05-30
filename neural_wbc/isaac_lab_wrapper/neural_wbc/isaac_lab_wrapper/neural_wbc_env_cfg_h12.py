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

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .rewards import NeuralWBCRewardCfg_H12
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [
            ".*shoulder_roll_link.*",
            ".*elbow.*link.*",
            ".*hand.*link",
            ".*torso_joint.*",
            ".*shoulder.*joint.*",
            ".*elbow.*joint.*",
        ],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": [".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder_roll_link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle_roll_link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
}

# If one is enabled, other will also be enabled
ENFORCED_TOGETHERNESS = {
    "left_shoulder_link": [".*left_shoulder.*link.*"],
    "right_shoulder_link": [".*right_shoulder.*link.*"],
    "left_shoulder_joint": [".*left_shoulder.*joint.*"],
    "right_shoulder_joint": [".*right_shoulder.*joint.*"],
    "root_velocity": ["root_.*velocity_*"],
    "root_orientation": ["root_.*orientation_*"],
}

H12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=get_data_path("motion_lib/h12.usd"),
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03),
        joint_pos={
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
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
            # armature=0.01,
            # friction=0.01,
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
            # armature=0.01,
            # friction=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            stiffness=0,
            damping=0,
            # armature=0.01,
            # friction=0.01,
        ),
    },
)


@configclass
class NeuralWBCEnvCfgH12(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 21
    observation_space = 993
    state_space = 1076

    # Distillation parameters:
    single_history_dim = 69
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    # OH2O mode is tracking the head and hand positions. This can be modified to train a different specialist
    # or use the full DISTILL_MASK_MODES_ALL to train a generalist policy.
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {
        "omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"],
        "h2o": DISTILL_MASK_MODES_ALL["h2o"],
        "exbody": DISTILL_MASK_MODES_ALL["exbody"],
        "humanplus": DISTILL_MASK_MODES_ALL["humanplus"],
    }
    # distill_mask_modes = {"exbody": DISTILL_MASK_MODES_ALL["exbody"]}

    enforced_mask_modes = {
        "left_shoulder_link": ENFORCED_TOGETHERNESS["left_shoulder_link"],
        "right_shoulder_link": ENFORCED_TOGETHERNESS["right_shoulder_link"],
        "left_shoulder_joint": ENFORCED_TOGETHERNESS["left_shoulder_joint"],
        "right_shoulder_joint": ENFORCED_TOGETHERNESS["right_shoulder_joint"],
        "root_velocity": ENFORCED_TOGETHERNESS["root_velocity"],
        "root_orientation": ENFORCED_TOGETHERNESS["root_orientation"],
    }

    robot: ArticulationCfg = H12_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
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
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # torso, shoulders, elbows

    base_name = "torso_link"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "torso_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.25, 0, 0], [0.25, 0, 0], [0, 0, 0.7]])

    # These are the bodies that are tracked by the teacher. They may also contain the extended
    # bodies.
    tracked_body_names = [
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
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

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

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "torso_link",
    ]

    undesired_contact_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ]

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    rewards = NeuralWBCRewardCfg_H12()

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.robot_name = "h12"
        self.reference_motion_manager.motion_path = get_data_path("motions/h12_test.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/h1_2_fitmotionONLY.xml")

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            # Do not reset mask when there is only one mode.
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            # self.distill_mask_sparsity_randomization_enabled = False
            # self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            # self.distill_mask_sparsity_randomization_enabled = True
            # self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
