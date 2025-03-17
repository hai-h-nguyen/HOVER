# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import time
from threading import RLock
from typing import Any

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class H1SDKWrapper:
    """This provides interface for unitree h1 robot."""

    def __init__(
        self,
        cfg: Any,
    ) -> None:
        """Initializes a new instance of the H1SDKWrapper class.

        Args:
            cfg (Any): The configuration object.
        """
        self.cfg = cfg
        self._low_cmd = unitree_go_msg_dds__LowCmd_()
        self._low_cmd_lock = RLock()
        self._cmd_publish_dt = self.cfg.cmd_publish_dt
        self._init_cmd()

        self._low_state = None
        self.crc = CRC()
        self._joint_positions = np.zeros(self.cfg.num_joints)
        self._joint_velocities = np.zeros(self.cfg.num_joints)
        self._torso_orientation_quat = np.array([1, 0, 0, 0])
        self._torso_angular_velocity = np.zeros(3)
        self._init_sdk()

        self._cmd_received = False
        self._cmd_publisher_thread_ptr = RecurrentThread(
            interval=self._cmd_publish_dt, target=self._cmd_publisher, name="control_loop"
        )
        self._cmd_publisher_thread_ptr.Start()

    def _cmd_publisher(self):
        """Publishes the low-level command to the SDK."""
        with self._low_cmd_lock:
            if not self._cmd_received:
                return
            self._lowcmd_publisher.Write(self._low_cmd)

    def _init_sdk(self):
        """Initializes the SDK for the H1 robot.
        This function initializes the SDK using the required configuration.

        Args:
            None
        """
        ChannelFactoryInitialize(0, self.cfg.network_interface)

        # Create publisher
        self._lowcmd_publisher = ChannelPublisher(self.cfg.command_channel, LowCmd_)
        self._lowcmd_publisher.Init()

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber(self.cfg.state_channel, LowState_)
        self.lowstate_subscriber.Init(self.state_handler, self.cfg.subscriber_freq)

    def _is_weak_motor(self, motor_idx: int) -> bool:
        """Check if a motor is a weak motor.

        Args:
            motor_idx (int): The index of the motor.

        Returns:
            bool: True if the motor is a weak motor, False otherwise.
        """
        return self.cfg.motor_id_to_name[motor_idx] in self.cfg.weak_motors

    def _is_motor_enabled(self, motor_id: int) -> bool:
        """Check if a motor is enabled.
        Args:
            motor_id (int): The ID of the motor.
        Returns:
            bool: True if the motor is enabled, False otherwise.
        """
        return self.cfg.motor_id_to_name[motor_id] in self.cfg.enabled_motors

    def publish_joint_position_cmd(self, cmd_joint_positions: np.ndarray):
        """Publishes joint position commands to the low-level command publisher.

        Args:
            cmd_joint_positions (np.ndarray): An array of joint positions to be published.
        """
        with self._low_cmd_lock:
            for joint_idx in range(self.cfg.num_joints):
                motor_idx = self.cfg.JointSeq2MotorID[joint_idx]
                self._low_cmd.motor_cmd[motor_idx].q = cmd_joint_positions[joint_idx]
                self._low_cmd.motor_cmd[motor_idx].dq = 0.0
                self._low_cmd.motor_cmd[motor_idx].tau = 0.0
            self._low_cmd.crc = self.crc.Crc(self._low_cmd)
            self._cmd_received = True

    def publish_joint_torque_cmd(self, cmd_joint_torques: np.ndarray):
        """Publishes joint torque commands to the low-level command publisher.

        Args:
            cmd_joint_torques (np.ndarray): An array of joint torques to be published.
        """
        with self._low_cmd_lock:
            for joint_idx in range(self.cfg.num_joints):
                motor_idx = self.cfg.JointSeq2MotorID[joint_idx]
                self._low_cmd.motor_cmd[motor_idx].q = 0.0
                self._low_cmd.motor_cmd[motor_idx].dq = 0.0
                self._low_cmd.motor_cmd[motor_idx].tau = cmd_joint_torques[joint_idx]
                self._low_cmd.motor_cmd[motor_idx].kp = 0.0
                self._low_cmd.motor_cmd[motor_idx].kd = 0.0
            self._low_cmd.crc = self.crc.Crc(self._low_cmd)
            self._cmd_received = True

    def reset(self, desired_joint_positions: np.ndarray | None = None) -> None:
        """Resets the robot to the given joint positions.

        Args:
            desired_joint_positions (np.ndarray | None, optional): An array of desired joint positions.
                Defaults to None: The robot will be reset to the 0 initial pose.
        """
        self.time_ = 0.0
        self.control_dt_ = self.cfg.reset_step_dt
        self.duration_ = self.cfg.reset_duration
        desired_joint_positions = desired_joint_positions.flatten()
        if desired_joint_positions is None:
            desired_joint_positions = np.zeros(self.cfg.num_joints)
        print("Resetting H1 to given pose.")
        while self.time_ < self.duration_:
            self.time_ += self.control_dt_
            ratio = self.time_ / self.duration_
            print(f"\rResetting: {int(self.duration_ - self.time_)}s remaining...", end="", flush=True)
            current_joint_positions = self.joint_positions
            target_joint_positions = (
                current_joint_positions + (desired_joint_positions - current_joint_positions) * ratio
            )
            self.publish_joint_position_cmd(target_joint_positions)
            time.sleep(self.control_dt_)
        print("\nReset complete.")

    def _init_cmd(self):
        """Initializes the low-level command.
        This function sets the values of the low-level command based on the configuration.
        """
        self._low_cmd.head[0] = self.cfg.head0
        self._low_cmd.head[1] = self.cfg.head1
        self._low_cmd.level_flag = self.cfg.level_flag
        self._low_cmd.gpio = self.cfg.gpio

        for i in range(len(self.cfg.motor_id_to_name)):
            if self._is_weak_motor(i):
                self._low_cmd.motor_cmd[i].mode = self.cfg.weak_motor_mode
                self._low_cmd.motor_cmd[i].kp = self.cfg.kp_low
                self._low_cmd.motor_cmd[i].kd = self.cfg.kd_low
            else:
                self._low_cmd.motor_cmd[i].mode = self.cfg.strong_motor_mode
                self._low_cmd.motor_cmd[i].kp = self.cfg.kp_high
                self._low_cmd.motor_cmd[i].kd = self.cfg.kd_high

            self._low_cmd.motor_cmd[i].q = 0
            self._low_cmd.motor_cmd[i].dq = 0
            self._low_cmd.motor_cmd[i].tau = 0

    @property
    def joint_positions(self) -> np.ndarray:
        """Returns the joint positions of the robot.

        Returns:
            np.ndarray: The joint positions.
        """
        return self._joint_positions

    @property
    def joint_velocities(self) -> np.ndarray:
        """Returns the joint velocities of the robot.

        Returns:
            np.ndarray: The joint velocities.
        """
        return self._joint_velocities

    @property
    def torso_orientation(self) -> np.ndarray:
        """Returns the torso orientation quaternion of the robot.
        The quaternion is in the format of [w, x, y, z].
        The orientation is in the world frame, which depends on the initial pose of the robot.

        Returns:
            np.ndarray: The torso orientation quaternion.
        """
        return self._torso_orientation_quat

    @property
    def torso_angular_velocity(self) -> np.ndarray:
        """Returns the torso angular velocity of the robot.
        The angular velocity is in robot frame, which does not depend on the initial pose of the robot.

        Returns:
            np.ndarray: The torso angular velocity.
        """
        return self._torso_angular_velocity

    def state_handler(self, msg: LowState_):
        """
        Update the joint positions and velocities based on the low state message.
        Saves them as per the joint sequence of isaac lab and mujoco.

        Args:
            msg (LowState_): The low state message containing the motor states.
        """
        self._low_state = msg
        # The orientation is in the world frame, which depends on the initial pose of the robot.
        self._torso_orientation_quat = self._low_state.imu_state.quaternion
        # The angular velocity is in robot frame, which does not depend on the initial pose of the robot.
        self._torso_angular_velocity = np.array(self._low_state.imu_state.gyroscope)
        for joint_idx in range(self.cfg.num_joints):
            motor_idx = self.cfg.JointSeq2MotorID[joint_idx]
            self._joint_positions[joint_idx] = msg.motor_state[motor_idx].q
            self._joint_velocities[joint_idx] = msg.motor_state[motor_idx].dq
