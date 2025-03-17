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

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Any, Literal

from hw_wrappers.h1_sdk_wrapper import H1SDKWrapper
from mujoco_wrapper.mujoco_simulator import WBCMujoco

from neural_wbc.core.robot_wrapper import Robot, register_robot


@register_robot
class UnitreeH1(Robot):
    """Real UniTree H1 robot, complimented with Mujoco for kinematics"""

    def __init__(
        self,
        cfg: Any,
        num_instances: int = 1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(cfg, num_instances=num_instances, device=device)
        self.cfg = cfg

        self._h1_sdk = H1SDKWrapper(cfg=cfg)
        self.device = device
        self.send_command = self._resolve_command_fn(robot_actuation_type=cfg.robot_actuation_type)

        self._kinematic_model = WBCMujoco(
            model_path=cfg.model_xml_path,
            sim_dt=cfg.dt,
            enable_viewer=cfg.enable_viewer,
            num_instances=num_instances,
            device=device,
        )

        # Assign gravity to mujoco model
        self._kinematic_model.model.opt.gravity = np.array([0, 0, self.cfg.gravity_value])

        self._joint_names = self._kinematic_model.joint_names
        self._body_names = self._kinematic_model.body_names

        self._free_joint_offset = 1 if self._kinematic_model.has_free_joint else 0

        self.joint_pos_offset = self._kinematic_model.joint_pos_offset
        self.joint_vel_offset = self._kinematic_model.joint_vel_offset

    def _resolve_command_fn(
        self,
        robot_actuation_type: Literal["Pos", "Torque"] = "Pos",
    ):
        if robot_actuation_type == "Pos":
            return self._send_position_command
        elif robot_actuation_type == "Torque":
            return self._send_torque_command
        else:
            raise ValueError(f"Unrecognized robot actuation type {robot_actuation_type}")

    def update(self, obs_dict: dict[str, torch.Tensor]) -> None:
        """Update the underlying model based on the observations from the environment/real robot.

        Args:
            obs_dict (dict[str, torch.Tensor]): A dictionary containing the latest robot observations.
        """
        # TODO (pulkitg): Change this to get data from vicon later on
        # For root use values from init state in config
        if "root_pos" in obs_dict:
            self._root_position = obs_dict["root_pos"]
        if "root_orientation" in obs_dict:
            self._root_rotation = obs_dict["root_orientation"]

        # We only need root position for the policy
        self._root_lin_vel = torch.zeros(1, 3).to(dtype=torch.float32, device=self.device)
        self._root_ang_vel = torch.zeros(1, 3).to(dtype=torch.float32, device=self.device)

        # Get state from the robot
        self._joint_positions = (
            torch.from_numpy(self._h1_sdk.joint_positions).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        )
        self._joint_velocities = (
            torch.from_numpy(self._h1_sdk.joint_velocities).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        )

        qpos = torch.hstack((self._root_position, self._root_rotation, self._joint_positions))
        qvel = torch.hstack((self._root_ang_vel, self._root_lin_vel, self._joint_velocities))

        self._kinematic_model.reset(qpos, qvel)

        self._joint_positions = self._kinematic_model.joint_positions
        self._joint_velocities = self._kinematic_model.joint_velocities
        self._body_positions = self._kinematic_model.body_positions
        self._body_rotations = self._kinematic_model.body_rotations
        self._body_lin_vels, self._body_ang_vels = self._kinematic_model.body_velocities

    def reset(self, **kwargs) -> None:
        """Resets the wrapper

        Args:
            kwargs (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.
        """
        qpos = kwargs.get("qpos")
        qvel = kwargs.get("qvel")
        self._kinematic_model.reset(qpos=qpos, qvel=qvel)

        joint_positions = qpos[..., self._kinematic_model.joint_pos_offset :]
        joint_positions_np = joint_positions.numpy()

        # Reset robot pose
        self._h1_sdk.reset(joint_positions_np)
        self.update({})

    def _send_torque_command(self, torques: np.ndarray | None = None) -> None:
        """Send torque commands to the robot"""
        if torques:
            self._h1_sdk.publish_joint_torque_cmd(torques.flatten())

    def _send_position_command(self, positions: np.ndarray | None = None) -> None:
        """Send position commands to the robot"""
        if positions is not None:
            self._h1_sdk.publish_joint_position_cmd(positions.flatten())

    def visualize(self, **payload) -> None:
        """Visualize some info

        Args:
            payload (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.

        """
        if "ref_motion_state" in payload:
            self._kinematic_model.visualize_ref_state(payload["ref_motion_state"])

    def step(self, actions: np.ndarray | None = None, nsteps: int = 1) -> None:
        """Step the simulation forward nsteps with the given action.

        Args:
            actions (np.ndarray | None, optional): Action to apply to the robot. Defaults to None.
            nsteps (int, optional): Number of steps to take. Defaults to 1.
        """
        self._kinematic_model.forward()
        self._kinematic_model.update_viewer()
        self.send_command(actions)

    def get_body_ids(self, body_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all bodies in the model, indexed after removing the world body.

        Args:
            body_names (list[str] | None, optional): Names of the bodies. Defaults to None.

        Returns:
            dict[str, int]: Mapping from body name to body id.
        """
        return self._kinematic_model.get_body_ids(body_names, self._free_joint_offset)

    def get_joint_ids(self, joint_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all joints in the model, indexed after removing the free joint.

        Args:
            joint_names (list[str] | None, optional): Names of the joints. Defaults to None.

        Returns:
            dict[str, int]: Mapping from joint name to joint id.
        """
        # TODO(pulkig): Resolve these with the unitree sdk motor IDs
        return self._kinematic_model.get_joint_ids(joint_names, self._free_joint_offset)

    def get_body_pose(self, body_name: str = "pelvis") -> tuple[torch.Tensor, torch.Tensor]:
        """Get the position and quaternion of the base

        Args:
            body_name (str, optional): Name of the body. Defaults to 'pelvis'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Position and quaternion of the base
        """
        # TODO(pulkitg): Change to use Vicon data or IMU
        return self._kinematic_model.get_body_pose(body_name)

    def get_base_projected_gravity(self, base_name: str = "torso_link") -> torch.Tensor:
        """Get the projection of the gravity vector to the base frame

        Args:
            base_name (str, optional): Name of the base. Defaults to 'torso_link'.

        Returns:
            torch.Tensor: Projection of the gravity vector to the base frame
        """
        # Get torso orientation in quaternion [w, x, y, z]
        torso_orientation_quat = self._h1_sdk.torso_orientation

        # Create rotation matrix from quaternion
        torso_rot_mat_np = R.from_quat(torso_orientation_quat, scalar_first=True).as_matrix()
        torso_rot_mat = torch.tensor(torso_rot_mat_np, device=self.device, dtype=torch.float32)

        # World gravity vector (normalized)
        world_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)

        # Transform gravity to base frame
        base_gravity = torso_rot_mat.T @ world_gravity

        return base_gravity.unsqueeze(0)

    def get_base_angular_velocity(self, base_name: str = "torso_link") -> torch.Tensor:
        """Get the angular velocity of the base

        Args:
            base_name (str, optional): Name of the base. Defaults to 'torso_link'.

        Returns:
            torch.Tensor: Angular velocity of the base
        """
        return torch.tensor(self._h1_sdk.torso_angular_velocity, device=self.device, dtype=torch.float32).unsqueeze(0)

    def get_terrain_heights(self) -> torch.Tensor:
        """Get the terrain height.

        Note: before the actual terrain height sensor needs to be added to the model, we assume they are just zeros.

        Returns:
            torch.Tensor: Terrain height
        """
        return self._kinematic_model.get_terrain_heights()
