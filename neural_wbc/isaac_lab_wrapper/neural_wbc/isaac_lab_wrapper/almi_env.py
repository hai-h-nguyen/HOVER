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

import copy
import h5py
import numpy as np
import pprint
import random
import torch
from typing import TYPE_CHECKING

import joblib
import pandas as pd

from neural_wbc.core.observations import StudentHistory
from neural_wbc.core.termination_almi import check_termination_conditions
from neural_wbc.isaac_lab_wrapper.almi_observations import compute_observations
from neural_wbc.isaac_lab_wrapper.control import resolve_control_fn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg, UniformNoiseCfg

from .body_state import build_body_state
from .rewards import NeuralWBCRewards_H12_ALMI
from .visualization import RefMotionVisualizer

if TYPE_CHECKING:
    from .env_cfg import NeuralWBCEnvCfg


def print_config(info: str, names: list[str], values: list | torch.Tensor):
    """Print named configuration values."""
    print(f"[INFO] {info}:")
    pprint.pprint(dict(zip(names, values)), indent=4, sort_dicts=False)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


class ALMIEnv(DirectRLEnv):
    cfg: NeuralWBCEnvCfg

    def __init__(self, cfg: NeuralWBCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = self.action_space.shape[1]
        self.num_observations = self.observation_space.shape[1]

        # Joint position command (deviation from default joint positions)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Resolve the joints on which the actions are applied.
        self._joint_ids, self._joint_names = self._robot.find_joints(
            self.cfg.joint_names,
            preserve_order=True,
        )
        print_config("self._joint_ids", self._joint_names, self._joint_ids)

        # Resolve the bodies.
        self._body_ids, self._body_names = self._robot.find_bodies(
            self.cfg.body_names,
            preserve_order=True,
        )
        print_config("self._body_ids", self._body_names, self._body_ids)

        # Get specific body indices
        self.base_id, self.base_name = self.contact_sensor.find_bodies(self.cfg.base_name)
        self.base_id = self.base_id[0]
        self.base_name = self.base_name[0]
        print_config("self._base_id", [self.base_name], [self.base_id])

        self.feet_ids, self.feet_names = self.contact_sensor.find_bodies(self.cfg.feet_name)
        print_config("self.feet_ids", self.feet_names, self.feet_ids)

        self._undesired_contact_body_ids, undesired_contact_body_name = self.contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names
        )
        print_config("self._undesired_contact_body_ids", undesired_contact_body_name, self._undesired_contact_body_ids)

        # resolve the pd gain for each joint
        self._p_gains = torch.zeros((self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device)
        self._d_gains = torch.zeros((self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device)

        self.kp_scale = torch.ones((self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device)
        self.kd_scale = torch.ones((self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device)

        self.default_kp_scale = self.kp_scale.clone()
        self.default_kd_scale = self.kd_scale.clone()

        for key, value in self.cfg.stiffness.items():
            joint_ids, joint_names = self._robot.find_joints(key, preserve_order=True)
            self._p_gains[:, joint_ids] = value
        for key, value in self.cfg.damping.items():
            joint_ids, joint_names = self._robot.find_joints(key, preserve_order=True)
            self._d_gains[:, joint_ids] = value
        self._p_gains = self._p_gains[:, self._joint_ids]
        self._d_gains = self._d_gains[:, self._joint_ids]

        print_config("self._p_gains", self._joint_names, self._p_gains[0])
        print_config("self._d_gains", self._joint_names, self._d_gains[0])

        # resolve the controller
        self._control_fn = resolve_control_fn(self.cfg.control_type)

        # resolve the control delay
        self.action_queue = torch.zeros(
            (self.num_envs, self.cfg.ctrl_delay_step_range[1] + 1, self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._action_delay = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (self.num_envs,),
            device=self.device,
            requires_grad=False,
        )

        # resolve the control noise: we will add noise to the final torques. the _rfi_lim defines
        # the sample range of the added noise. It represented by the percentage of the control limits.
        # noise = uniform(self.rfi_lim*joint_effort_limit, self.rfi_lim_*joint_effort_limit)
        self.default_rfi_lim = self.cfg.default_rfi_lim * torch.ones(
            (self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device
        )
        self.rfi_lim = self.default_rfi_lim.clone()

        # resolve torque_limits (the order here is IsaacLab order, have not reordered by cfg.)
        self.torque_limits = torch.ones((self.num_envs, 21), dtype=torch.float, device=self.device)
        for actuator in self._robot.actuators.values():
            actuator_joint_ids = actuator.joint_indices
            actuator_torque_limits = actuator.effort_limit
            self.torque_limits[:, actuator_joint_ids] *= actuator_torque_limits
        print("[INFO]: self.torque_limits", self._robot.joint_names, self.torque_limits[0])

        # Randomize robot base com
        if not hasattr(self, "default_coms"):
            self.default_coms = self._robot.root_physx_view.get_coms().clone()
            self.base_com_bias = torch.zeros((self.num_envs, 3), dtype=torch.float, device="cpu")

        # Initialize recovery counters
        self.recovery_counters = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.add_policy_obs_noise:
            self._policy_observation_noise_scale_vec = self._get_policy_observation_noise_scale()
            print(
                "[INFO]: self._policy_observation_noise_scale_vec",
                self._policy_observation_noise_scale_vec,
                self._policy_observation_noise_scale_vec.shape,
            )
            observation_noise_model_cfg = NoiseModelCfg(
                noise_cfg=UniformNoiseCfg(
                    n_min=-self._policy_observation_noise_scale_vec,
                    n_max=self._policy_observation_noise_scale_vec,
                    operation="add",
                )
            )
            self._observation_noise_model: NoiseModel = observation_noise_model_cfg.class_type(
                observation_noise_model_cfg, num_envs=self.num_envs, device=self.device
            )

        # Cache body mass randomization scale for privileged observation.
        if not hasattr(self, "body_mass_scale"):
            self.mass_randomized_body_ids, _ = self._robot.find_bodies(
                self.cfg.mass_randomized_body_names, preserve_order=True
            )
            self.body_mass_scale = torch.ones(
                self.num_envs, len(self.mass_randomized_body_ids), dtype=torch.float, device=self.device
            )

        # Logging reward sums
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in self.cfg.rewards.scales
        }

        self.ref_episodic_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_ref_episodic_offset = self.ref_episodic_offset.clone()

        # Start positions of each environment
        self._start_positions_on_terrain = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        # Rewards
        feet_body_ids = [self._body_names.index(feet_name) for feet_name in self.feet_names]
        self._rewards = NeuralWBCRewards_H12_ALMI(
            env=self,
            reward_cfg=self.cfg.rewards,
            contact_sensor=self.contact_sensor,
            contact_sensor_feet_ids=self.feet_ids,
            body_state_feet_ids=feet_body_ids,
            root_ids=self.cfg.root_id,
        )
        self._termination_conditions = {}

        # Curriculum
        self.average_episode_length = 0
        self.penalty_scale = 1.0

        # Commands
        self.commands = torch.zeros(
            self.num_envs, self.cfg.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_scales = torch.tensor(
            [self.cfg.obs_scales["lin_vel"], self.cfg.obs_scales["lin_vel"], self.cfg.obs_scales["ang_vel"]],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.commands_range = self.cfg.commands_range

        self.arm_weight = self.cfg.init_arm_weight
        if self.cfg.arm_curriculum:
            self.motion_weight = self.cfg.init_motion_weight
            self.motion_leading = self.cfg.motion_leading
            self.motion_weight_increase = self.cfg.motion_weight_increase
            self.motion_weight_decrease = self.cfg.motion_weight_decrease
            self.arm_weight_increase = self.cfg.arm_weight_increase
            self.arm_weight_decrease = self.cfg.arm_weight_decrease
            self.motion_range = self.cfg.motion_range

        # Load motion
        self._load_motion()

        self.default_joint_pos = torch.tensor(
            [
                0.0,
                -0.4,
                0.0,
                0.8,
                -0.4,
                0.0,
                0.0,
                -0.4,
                0.0,
                0.8,
                -0.4,
                0.0,
                0.0,
                0.4,
                0.2,
                0.0,
                0.3,
                0.4,
                -0.2,
                0.0,
                0.3,
            ],
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self._previous_joint_vel = torch.zeros(
            (self.num_envs, len(self.cfg.joint_names)), dtype=torch.float, device=self.device, requires_grad=False
        )

    def _update_arm_curriculum(self, env_ids):
        mean_episode_length = torch.sum(self.episode_length_buf[env_ids]) / len(env_ids)
        episode_length_ratio = mean_episode_length / self.max_episode_length

        leading_weight, other_weight = self._cal_motion_and_arm_weight(episode_length_ratio)
        self.motion_weight = leading_weight if self.motion_leading else other_weight
        self.arm_weight = other_weight if self.motion_leading else leading_weight

    def _cal_motion_and_arm_weight(self, episode_length_ratio):
        leading_weight = self.motion_weight if self.motion_leading else self.arm_weight
        other_weight = self.arm_weight if self.motion_leading else self.motion_weight

        leading_increase = self.motion_weight_increase if self.motion_leading else self.arm_weight_increase
        leading_decrease = self.motion_weight_decrease if self.motion_leading else self.arm_weight_decrease
        other_increase = self.arm_weight_increase if self.motion_leading else self.motion_weight_increase
        other_decrease = self.arm_weight_decrease if self.motion_leading else self.motion_weight_decrease

        if episode_length_ratio > 0.8:
            other_weight += other_increase  # 0.01
        else:
            other_weight -= other_decrease  # 0.1

        if other_weight > 1:
            other_weight = 0
            leading_weight += leading_increase  # 0.05
            if leading_weight > 1:
                leading_weight = 1
                other_weight = 1

        if other_weight < 0:
            other_weight = 1 + other_weight
            leading_weight -= leading_decrease
            if leading_weight < 0:
                leading_weight = 0
                other_weight = 0
        return leading_weight, other_weight

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.commands_range["lin_vel_x"][0],
            self.commands_range["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.commands_range["lin_vel_y"][0],
            self.commands_range["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.commands_range["heading"][0],
                self.commands_range["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.commands_range["ang_vel_yaw"][0],
                self.commands_range["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self._episode_sums["reward_track_lin_vel"][env_ids]) / self.max_episode_length > 0.8:
            self.commands_range["lin_vel_x"][0] = np.clip(
                self.commands_range["lin_vel_x"][0] - 0.5, -self.cfg.max_commands_curriculum, 0.0
            )
            self.commands_range["lin_vel_x"][1] = np.clip(
                self.commands_range["lin_vel_x"][1] + 0.5, 0.0, self.cfg.max_commands_curriculum
            )

    def _alloc_motion(self, env_ids):
        if self.cfg.arm_curriculum:
            random_lower = max(0, int((self.motion_weight - self.motion_range) * len(self.new_motion)))
            random_upper = min(
                len(self.new_motion) - 1, int((self.motion_weight + self.motion_range) * len(self.new_motion))
            )
            random_indices = torch.randint(random_lower, random_upper + 1, (len(env_ids),), device=self.device)
            for i in range(len(env_ids)):
                self.motion_buffer[int(env_ids[i])] = torch.stack(
                    self.new_motion[random_indices[i]][: self.motion_length]
                )
                # self.motion_buffer[int(env_ids[i])] = torch.stack(self.new_motion[int(env_ids[i])][:self.motion_length])
                self.env_motion_dict[int(env_ids[i])] = random_indices[i]
        else:
            random_indices = torch.randint(0, len(self.new_motion), (self.num_envs,), device=self.device)
            for i in range(len(env_ids)):
                self.motion_buffer[int(env_ids[i])] = torch.stack(
                    self.new_motion[random_indices[i]][: self.motion_length]
                )
                self.env_motion_dict[int(env_ids[i])] = random_indices[i]

    def _load_motion(self):
        print(f"Load motion from {self.cfg.motion_path}......")
        motion_data = joblib.load(self.cfg.motion_path)  # shape: (num_motion, num_frames, num_dof)

        df = pd.read_csv(self.cfg.mean_episode_length_path)
        if len(df) == len(motion_data):
            sorted_indices = df.sort_values("mean_episode_length", ascending=False)["env_id"].tolist()
            # sorted_indices = [i for i in range(len(motion_data))]
            motion_data = [motion_data[i] for i in sorted_indices]
        else:
            sorted_indices = [i for i in range(len(motion_data))]
            random.shuffle(sorted_indices)
            motion_data = [motion_data[i] for i in sorted_indices]

        self.motion_data = motion_data
        new_motion = []
        self.env_motion_dict = {}

        for i in range(len(motion_data)):
            if len(motion_data[i]) < 1002:
                last_row = motion_data[i][-1][:, -9:]
                padding = [torch.from_numpy(last_row).squeeze() for _ in range(1002 - len(motion_data[i]))]

                repeated_frames = []
                for motion_frame in motion_data[i]:
                    frame = torch.from_numpy(motion_frame[:, -9:]).squeeze()
                    repeated_frames.extend([frame, frame])

                repeated_padding = []
                for pad in padding:
                    repeated_padding.extend([pad, pad])
                new_motion.append(repeated_frames + repeated_padding)
            else:
                repeated_frames = []
                for motion_frame in motion_data[i][:1002]:
                    frame = torch.from_numpy(motion_frame[:, -9:]).squeeze()
                    repeated_frames.extend([frame, frame])  # Repeat each frame twice
                new_motion.append(repeated_frames)
        self.new_motion = new_motion

        # shape: (num_envs, max_episode_length+1, num_arm_dof)
        motion_length = int(self.max_episode_length + 1)
        self.motion_length = motion_length
        self.motion_buffer = torch.zeros(
            (self.num_envs, motion_length, 9), dtype=torch.float, device=self.device, requires_grad=False
        )

        temp_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._alloc_motion(temp_env_ids)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Update recovery counters
        self.recovery_counters -= 1
        self.recovery_counters = self.recovery_counters.clip(min=0)

        actions = torch.clip(actions, -100.0, 100.0).to(self.device)

        # Action delay process
        if self.cfg.ctrl_delay_step_range[1] > 0:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            self.actions = self.action_queue[torch.arange(self.num_envs), self._action_delay].clone()
        else:
            self.actions = actions.clone()

    def _apply_action(self):
        # We define our own control law that can update at every physics step. Therefore we put
        # this computation here.
        actions_scaled = self.actions * self.cfg.action_scale
        arm_action = (
            self.motion_buffer[torch.arange(self.num_envs), self.episode_length_buf] - self.default_joint_pos[-9:]
        ) * (self.arm_weight if self.cfg.arm_curriculum else 1)
        arm_action[:, 0] = 0.0  # set torso joint to 0.0
        wholebody_action = torch.cat([actions_scaled, arm_action], dim=-1)
        self._processed_actions = self._control_fn(self, wholebody_action, joint_ids=self._joint_ids)

        # Adding noise to the control signal to enhance robustness. `torque_limits` is using IsaacLab
        # ordering, therefore we need to reorder the noise.
        actions_noise = (
            (torch.rand_like(self._processed_actions) * 2.0 - 1.0)
            * self.rfi_lim
            * self.torque_limits[:, self._joint_ids]
        )
        self._processed_actions += actions_noise

        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        period = 0.8
        self.is_stance_threshold = 0.55
        self.offset = torch.where(torch.norm(self.commands[:, :3], dim=1) < 0.1, 0, 0.5)

        self.phase = torch.where(
            torch.norm(self.commands[:, :3], dim=1) < 0.1,
            0,
            (self.episode_length_buf * self.cfg.decimation * self.cfg.dt) % period / period,
        )
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.offset) % 1

        self.left_sin_phase = torch.sin(2 * np.pi * self.phase_left).unsqueeze(1)
        self.right_sin_phase = torch.sin(2 * np.pi * self.phase_right).unsqueeze(1)

        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        self._previous_actions = self.actions.clone()

        body_state = build_body_state(
            data=self._robot.data,
            root_id=self.cfg.root_id,
            body_ids=self._body_ids,
            joint_ids=self._joint_ids,
            extend_body_parent_ids=None,
            extend_body_pos=None,
        )

        # Add additional data that is useful for evaluation.
        self.extras["data"] = {
            "state": {
                "body_pos": body_state.body_pos_extend.detach().clone(),
                "joint_pos": body_state.joint_pos.detach().clone(),
                "root_pos": body_state.root_pos.detach().clone(),
                "root_lin_vel": body_state.root_lin_vel.detach().clone(),
                "root_rot": body_state.root_rot.detach().clone(),
            },
            "upper_joint_ids": self.cfg.upper_body_joint_ids,
            "lower_joint_ids": self.cfg.lower_body_joint_ids,
        }

        obs_dic = compute_observations(
            env=self,
            body_state=body_state,
        )

        if self.cfg.add_policy_obs_noise:
            obs_dic["teacher_policy"] = self._observation_noise_model.apply(obs_dic["teacher_policy"])

        self._previous_joint_vel[:] = body_state.joint_vel[:]

        return obs_dic

    def _get_rewards(self) -> torch.Tensor:
        body_state = build_body_state(
            data=self._robot.data,
            root_id=self.cfg.root_id,
            body_ids=self._body_ids,
            joint_ids=self._joint_ids,
            extend_body_parent_ids=None,
            extend_body_pos=None,
        )

        timeout_buf = self.reset_time_outs

        reward_sum, rewards = self._rewards.compute_reward(
            body_state=body_state,
            articulation_data=self._robot.data,
            previous_actions=self._previous_actions,
            actions=self.actions,
            reset_buf=self.reset_buf,  # From DirectRLEnv
            timeout_buf=timeout_buf,
            penalty_scale=self.penalty_scale,
            processed_actions=self._processed_actions.detach().clone(),
            previous_joint_velocities=self._previous_joint_vel.detach().clone(),
            commands=self.commands,
            leg_phases=self.leg_phase.detach().clone(),
        )

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward_sum

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Post physics step callback
        env_ids = (
            (self.episode_length_buf % int(self.cfg.resampling_time / (self.cfg.dt * self.cfg.decimation)) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.commands_range["ang_vel_yaw"][0],
                self.commands_range["ang_vel_yaw"][1],
            )

        # period = 0.8
        # self.is_stance_threshold = 0.55
        # self.offset = torch.where(torch.norm(self.commands[:, :3], dim=1) < 0.1, 0, 0.5)

        # self.phase = torch.where(torch.norm(self.commands[:, :3], dim=1) < 0.1, 0, (self.episode_length_buf * self.cfg.decimation * self.cfg.dt) % period / period)
        # self.phase_left = self.phase
        # self.phase_right = (self.phase + self.offset) % 1

        # self.left_sin_phase = torch.sin(2 * np.pi * self.phase_left).unsqueeze(1)
        # self.right_sin_phase = torch.sin(2 * np.pi * self.phase_right).unsqueeze(1)

        # self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # Get dones
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died, self._termination_conditions = check_termination_conditions(
            projected_gravity=self._robot.data.projected_gravity_b,
            gravity_x_threshold=self.cfg.gravity_x_threshold,
            gravity_y_threshold=self.cfg.gravity_y_threshold,
            # We only need the last net contact forces.
            net_contact_forces=self.contact_sensor.data.net_forces_w_history[:, 0, :, :],
            undesired_contact_body_ids=self._undesired_contact_body_ids,
        )

        self.extras["termination_conditions"] = self._termination_conditions

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands_curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self._update_command_curriculum(env_ids)

        # update arm_weight curriculum
        if self.cfg.arm_curriculum:
            self._update_arm_curriculum(env_ids)

        self._resample_commands(env_ids)

        self._alloc_motion(env_ids)

        # reset actions
        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_joint_vel[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self._action_delay[env_ids] = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (len(env_ids),),
            device=self.device,
            requires_grad=False,
        )

        # reset recovery counter
        self.recovery_counters[env_ids] = 0.0

        # Logging
        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0 * self._episode_sums[key][env_ids]

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # Utility functions
    def _get_episode_times(self):
        return self.episode_length_buf * self.cfg.decimation * self.cfg.dt

    def _get_policy_observation_noise_scale(self):
        noise_vec = torch.zeros(self.num_observations, device=self.device)
        noise_vec[:3] = (
            self.cfg.policy_obs_noise_scales["ang_vel"]
            * self.cfg.policy_obs_noise_level
            * self.cfg.obs_scales["ang_vel"]
        )
        noise_vec[3:6] = self.cfg.policy_obs_noise_scales["gravity"] * self.cfg.policy_obs_noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9:30] = (
            self.cfg.policy_obs_noise_scales["dof_pos"]
            * self.cfg.policy_obs_noise_level
            * self.cfg.obs_scales["dof_pos"]
        )
        noise_vec[30:51] = (
            self.cfg.policy_obs_noise_scales["dof_vel"]
            * self.cfg.policy_obs_noise_level
            * self.cfg.obs_scales["dof_vel"]
        )
        noise_vec[51:63] = 0.0  # previous actions
        noise_vec[63:65] = 0.0  # sin/cos phase
        return noise_vec

    def get_terrain_heights(self) -> torch.Tensor:
        return self._height_scanner.data.ray_hits_w.clone()[..., 2]
