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

import math
import numpy as np
import time

# from tf2_msgs.msg import TFMessage
import xml.etree.ElementTree as ET

# import time
from datetime import datetime
from threading import RLock
from typing import Any, List

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    geometry_msgs_msg_dds__ListTransformStamped_,
    geometry_msgs_msg_dds__TransformStamped_,
    unitree_hg_msg_dds__LogCmd_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import ListTransformStamped_, TransformStamped_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LogCmd_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from neural_wbc.core.util import Filter

WAIST_PITCH_MOTOR_ID = 14
WAIST_ROLL_MOTOR_ID = 13


class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle representation to quaternion."""
    # Handle zero rotation case
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    # Normalize axis
    axis = axis_angle / angle

    # Calculate quaternion components
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    return {
        "x": float(axis[0] * sin_half),
        "y": float(axis[1] * sin_half),
        "z": float(axis[2] * sin_half),
        "w": float(cos_half),
    }


def parse_frame_map(file_path):
    tree = ET.parse(file_path)
    frame_map = {}

    for i in tree.findall("joint"):
        current_joint = i.attrib["name"]

        parent = i.find("parent")
        parent_joint = parent.attrib["link"]

        child = i.find("child")
        child_joint = child.attrib["link"]

        trans = {"x": 0, "y": 0, "z": 0}
        origin = i.find("origin")
        if origin is not None:
            trans_data = origin.attrib["xyz"].split()
            trans = {
                "x": float(trans_data[0]),
                "y": float(trans_data[1]),
                "z": float(trans_data[2]),
            }

        frame_map[current_joint] = {
            "parent": parent_joint,
            "child": child_joint,
            "translation": trans,
        }

    return frame_map


G1_DOF_AXIS = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
)


class G1SDKWrapper:
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
        self.frame_map = parse_frame_map(self.cfg.model_urdf_path)
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._log_cmd = unitree_hg_msg_dds__LogCmd_()
        self._transform_cmd = geometry_msgs_msg_dds__TransformStamped_()
        self._list_transform_cmd = geometry_msgs_msg_dds__ListTransformStamped_()
        self._low_cmd_lock = RLock()
        self._cmd_publish_dt = self.cfg.cmd_publish_dt

        self._update_mode_machine = False

        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

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
        print("G1 SDK Wrapper initialized.")

        self.first_time = 1
        self.filter = Filter(T1=0.005, Ts=0.02, num_dofs=self.cfg.num_joints)
        self.reference_joint_pos = [0.0] * self.cfg.num_joints
        self.reference_joint_vel = [0.0] * self.cfg.num_joints
        self.reference_root_pos = [0.0] * 3
        self.reference_root_rot = [0.0] * 4

    def _cmd_publisher(self):
        """Publishes the low-level command to the SDK."""
        with self._low_cmd_lock:
            if not self._cmd_received:
                return
            self._lowcmd_publisher.Write(self._low_cmd)
            self._logcmd_publisher.Write(self._log_cmd)
            # self._listtransformcmd_publisher.Write(self._list_transform_cmd)

    def _init_sdk(self):
        """Initializes the SDK for the H1 robot.
        This function initializes the SDK using the required configuration.

        Args:
            None
        """
        if self.cfg.network_interface == "lo":
            ChannelFactoryInitialize(1, self.cfg.network_interface)
            print("Using local interface.")
        else:
            ChannelFactoryInitialize(0, self.cfg.network_interface)
            print(f"Using network interface: {self.cfg.network_interface}")

        # Create publisher
        self._lowcmd_publisher = ChannelPublisher(self.cfg.command_channel, LowCmd_)
        self._lowcmd_publisher.Init()

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber(self.cfg.state_channel, LowState_)
        self.lowstate_subscriber.Init(self.state_handler, self.cfg.subscriber_freq)

        # Create information publisher
        self._logcmd_publisher = ChannelPublisher(self.cfg.information_channel, LogCmd_)
        self._logcmd_publisher.Init()

        # Create visualizer reference motion publisher
        # self._listtransformcmd_publisher = ChannelPublisher(self.cfg.vis_ref_motion_channel, ListTransformStamped_)
        # self._listtransformcmd_publisher.Init()

        print("Waiting for the robot to connect...")
        self.init_cmd_hg(self._low_cmd, self.mode_machine_, self.mode_pr_)  # type: ignore
        self.wait_for_low_state()
        self.init_log_hg(self._log_cmd)
        # self.init_list_tf_cmd(self._list_transform_cmd, self._transform_cmd)

    def wait_for_low_state(self):
        while not self._update_mode_machine:
            time.sleep(0.02)
        print("Successfully connected to the robot.")

    def init_cmd_hg(self, cmd: LowCmd_, mode_machine: int, mode_pr: int):
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        for i, motor_name in ({**self.cfg.motor_id_to_name, **self.cfg.wrist_motor_id_to_name}).items():
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].qd = 0.0
            cmd.motor_cmd[i].kp = self.cfg.stiffness[motor_name + "_joint"] * 0.5
            cmd.motor_cmd[i].kd = self.cfg.damping[motor_name + "_joint"] * 2.0
            cmd.motor_cmd[i].tau = 0.0
        # cmd.motor_cmd[WAIST_PITCH_MOTOR_ID].mode = 0
        # cmd.motor_cmd[WAIST_PITCH_MOTOR_ID].kp = 0.
        # cmd.motor_cmd[WAIST_PITCH_MOTOR_ID].kd = 0.
        # cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].tau = 0.

        # cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].mode = 0
        # cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].kp = 0.
        # cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].kd = 0.
        # cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].tau = 0.

    def init_log_hg(self, logcmd: LogCmd_):
        for i, motor_name in ({**self.cfg.motor_id_to_name, **self.cfg.wrist_motor_id_to_name}).items():
            logcmd.motor_cmd[i].mode = 1
            logcmd.motor_cmd[i].q = 0.0
            logcmd.motor_cmd[i].qd = 0.0
            logcmd.motor_cmd[i].kp = self.cfg.stiffness[motor_name + "_joint"] * 0.5
            logcmd.motor_cmd[i].kd = self.cfg.damping[motor_name + "_joint"] * 1.5
            logcmd.motor_cmd[i].tau = 0.0

        logcmd.motor_state = self._low_state.motor_state
        logcmd.imu_state = self._low_state.imu_state

    def init_list_tf_cmd(self, _list_transform_cmd: ListTransformStamped_, _transform_cmd: TransformStamped_):
        axis_angle = list(self.cfg.robot_init_state["joint_pos"].values())
        time = datetime.now().timestamp()
        time_sec = int(time // 1_000_000_000)
        time_nano = int(time % 1_000_000_000)
        for joint_idx in range(len(self.cfg.joint_names)):
            axis_angle_2_euler = G1_DOF_AXIS[joint_idx] * axis_angle[joint_idx]
            quat_angle = axis_angle_to_quaternion(axis_angle_2_euler)

            joint_name = self.cfg.joint_names[joint_idx]
            joint_info = self.frame_map.get(joint_name, None)
            if not joint_info:
                print(joint_name)
                continue
            parent_frame_id = joint_info["parent"]
            child_frame_id = joint_info["child"]
            translation = joint_info["translation"]

            _transform_cmd.header.stamp.sec = time_sec
            _transform_cmd.header.stamp.nanosec = time_nano
            _transform_cmd.header.frame_id = parent_frame_id
            _transform_cmd.child_frame_id = child_frame_id
            _transform_cmd.transform.translation.x = translation["x"]
            _transform_cmd.transform.translation.y = translation["y"]
            _transform_cmd.transform.translation.z = translation["z"]
            _transform_cmd.transform.rotation.x = quat_angle["x"]
            _transform_cmd.transform.rotation.y = quat_angle["y"]
            _transform_cmd.transform.rotation.z = quat_angle["z"]
            _transform_cmd.transform.rotation.w = quat_angle["w"]
            _list_transform_cmd.data[joint_idx] = _transform_cmd
        _transform_cmd.header.stamp.sec = time_sec
        _transform_cmd.header.stamp.nanosec = time_nano
        _transform_cmd.header.frame_id = "world"
        _transform_cmd.child_frame_id = "pelvis"
        _transform_cmd.transform.translation.x = self.cfg.robot_init_state["base_pos"][0]
        _transform_cmd.transform.translation.y = self.cfg.robot_init_state["base_pos"][1]
        _transform_cmd.transform.translation.z = self.cfg.robot_init_state["base_pos"][2]
        _transform_cmd.transform.rotation.x = self.cfg.robot_init_state["base_quat"][0]
        _transform_cmd.transform.rotation.y = self.cfg.robot_init_state["base_quat"][1]
        _transform_cmd.transform.rotation.z = self.cfg.robot_init_state["base_quat"][2]
        _transform_cmd.transform.rotation.w = self.cfg.robot_init_state["base_quat"][3]
        _list_transform_cmd.data[-1] = _transform_cmd
        # print(_list_transform_cmd)
        # exit()

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
        # RESET SIM STATE BEFORE RUNNING POLICY, trigger by sending mode 0 to the first motor
        # if self.first_time > 0:
        #     self.first_time -=1
        #     self._low_cmd.motor_cmd[0].mode = 0
        # else:
        #     self._low_cmd.motor_cmd[0].mode = 1

        # compute desired joint velocities
        # self.desired_velocities = self.filter.dt1_filter(cmd_joint_positions)
        # self.desired_velocities = np.clip(self.desired_velocities, -32., 32.)

        with self._low_cmd_lock:
            for joint_idx in range(self.cfg.num_joints):
                motor_idx = self.cfg.JointSeq2MotorID[joint_idx]
                self._low_cmd.motor_cmd[motor_idx].q = cmd_joint_positions[joint_idx]
                # self._low_cmd.motor_cmd[motor_idx].dq = self.cmd_joint_velocities[joint_idx] if not math.isnan(self.cmd_joint_velocities[joint_idx]) else 0.0
                self._low_cmd.motor_cmd[motor_idx].dq = 0.0
                self._low_cmd.motor_cmd[motor_idx].tau = 0.0

                self._log_cmd.motor_cmd[motor_idx].q = cmd_joint_positions[joint_idx]
                self._log_cmd.motor_cmd[motor_idx].dq = 0.0
                self._log_cmd.motor_cmd[motor_idx].tau = 0.0
                self._log_cmd.reference_joint_pos[motor_idx] = self.reference_joint_pos[joint_idx]
                self._log_cmd.reference_joint_vel[motor_idx] = self.reference_joint_vel[joint_idx]

            self._log_cmd.motor_state = self._low_state.motor_state
            self._log_cmd.imu_state = self._low_state.imu_state

            # for i in range(12):
            #     motor_idx = self.cfg.JointSeq2MotorID[i]
            #     self._low_cmd.motor_cmd[motor_idx].q = 0.0
            #     self._low_cmd.motor_cmd[motor_idx].dq = 0.0
            #     self._low_cmd.motor_cmd[motor_idx].tau = 0.0
            # time = datetime.now().timestamp()
            # time_sec = int(time // 1_000_000_000)
            # time_nano = int(time % 1_000_000_000)
            # for joint_idx in range(len(self.cfg.joint_names)):
            #     axis_angle = self.reference_joint_pos[joint_idx]
            #     axis_angle_2_euler = G1_DOF_AXIS[joint_idx] * axis_angle
            #     quat_angle = axis_angle_to_quaternion(axis_angle_2_euler)

            #     joint_name = self.cfg.joint_names[joint_idx]
            #     joint_info = self.frame_map.get(joint_name, None)
            #     if not joint_info:
            #         continue
            #     parent_frame_id = joint_info["parent"]
            #     child_frame_id = joint_info["child"]
            #     translation = joint_info["translation"]

            #     self._transform_cmd.header.stamp.sec = time_sec
            #     self._transform_cmd.header.stamp.nanosec = time_nano
            #     self._transform_cmd.header.frame_id = parent_frame_id
            #     self._transform_cmd.child_frame_id = child_frame_id
            #     self._transform_cmd.transform.translation.x = translation["x"]
            #     self._transform_cmd.transform.translation.y = translation["y"]
            #     self._transform_cmd.transform.translation.z = translation["z"]
            #     self._transform_cmd.transform.rotation.x = quat_angle["x"]
            #     self._transform_cmd.transform.rotation.y = quat_angle["y"]
            #     self._transform_cmd.transform.rotation.z = quat_angle["z"]
            #     self._transform_cmd.transform.rotation.w = quat_angle["w"]
            #     self._list_transform_cmd.data[joint_idx] = self._transform_cmd
            # self._transform_cmd.header.stamp.sec = time_sec
            # self._transform_cmd.header.stamp.nanosec = time_nano
            # self._transform_cmd.header.frame_id = "world"
            # self._transform_cmd.child = "pelvis"
            # self._transform_cmd.transform.translation.x = self.reference_root_pos[0]
            # self._transform_cmd.transform.translation.y = self.reference_root_pos[1]
            # self._transform_cmd.transform.translation.z = self.reference_root_pos[2]
            # self._transform_cmd.transform.rotation.x = self.reference_root_rot[0]
            # self._transform_cmd.transform.rotation.y = self.reference_root_rot[1]
            # self._transform_cmd.transform.rotation.z = self.reference_root_rot[2]
            # self._transform_cmd.transform.rotation.w = self.reference_root_rot[3]
            # self._list_transform_cmd.data[-1] = self._transform_cmd

            self._low_cmd.crc = self.crc.Crc(self._low_cmd)
            self._cmd_received = True

            # print(self._list_transform_cmd)

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
            self._low_cmd.motor_cmd[WAIST_PITCH_MOTOR_ID].tau = 0.0
            self._low_cmd.motor_cmd[WAIST_ROLL_MOTOR_ID].tau = 0.0
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

        self.desired_joint_positions_init = np.array(list(self.cfg.robot_init_state["joint_pos"].values()))

        print("Resetting G1 to default pose.")
        print("desired_joint_positions: ", desired_joint_positions)
        while self.time_ < self.duration_:
            self.time_ += self.control_dt_
            ratio = self.time_ / self.duration_
            print(f"\rResetting: {int(self.duration_ - self.time_)}s remaining...", end="", flush=True)
            current_joint_positions = self.joint_positions
            target_joint_positions = (
                current_joint_positions + (self.desired_joint_positions_init - current_joint_positions) * ratio
            )
            self.publish_joint_position_cmd(target_joint_positions)
            # print("current_joint_positions: ", current_joint_positions[0:15])
            time.sleep(self.control_dt_)

        if desired_joint_positions is None:
            desired_joint_positions = np.zeros(self.cfg.num_joints)
        self.time_ = 0.0

        print("Resetting G1 to initial pose.")
        print("desired_joint_positions: ", desired_joint_positions)
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
        if self._update_mode_machine == False:
            self._update_mode_machine = True
        self._low_state = msg
        self.mode_machine_ = self._low_state.mode_machine
        # The orientation is in the world frame, which depends on the initial pose of the robot.
        self._torso_orientation_quat = self._low_state.imu_state.quaternion
        # The angular velocity is in robot frame, which does not depend on the initial pose of the robot.
        self._torso_angular_velocity = np.array(self._low_state.imu_state.gyroscope)
        for joint_idx in range(self.cfg.num_joints):
            motor_idx = self.cfg.JointSeq2MotorID[joint_idx]
            self._joint_positions[joint_idx] = msg.motor_state[motor_idx].q
            self._joint_velocities[joint_idx] = msg.motor_state[motor_idx].dq
