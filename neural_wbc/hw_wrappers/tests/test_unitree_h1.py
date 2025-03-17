# SPDX-FileCopyrightText: Copyright (c) 2024-25 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import unittest
from scipy.spatial.transform import Rotation as R
from unittest.mock import MagicMock, patch

from hw_wrappers.unitree_h1 import UnitreeH1


class TestUnitreeH1(unittest.TestCase):
    def setUp(self):
        # Mock the config
        self.cfg = MagicMock()
        self.cfg.robot_actuation_type = "Pos"
        self.cfg.dt = 0.01
        self.cfg.enable_viewer = False
        self.cfg.gravity_value = -9.81
        self.cfg.model_xml_path = "path/to/model.xml"

        # Create a mock H1SDKWrapper
        self.h1_sdk_patcher = patch("hw_wrappers.unitree_h1.H1SDKWrapper")
        self.mock_h1_sdk = self.h1_sdk_patcher.start()

        # Create a mock WBCMujoco
        self.mujoco_patcher = patch("hw_wrappers.unitree_h1.WBCMujoco")
        self.mock_mujoco = self.mujoco_patcher.start()

        # Initialize robot with CPU device for testing
        self.device = torch.device("cpu")
        self.robot = UnitreeH1(self.cfg, device=self.device)

    def tearDown(self):
        self.h1_sdk_patcher.stop()
        self.mujoco_patcher.stop()

    def test_get_base_projected_gravity_upright(self):
        """Test gravity projection when robot is upright"""
        # Set upright orientation (identity quaternion)
        self.robot._h1_sdk.torso_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # Call the function
        gravity = self.robot.get_base_projected_gravity("pelvis")

        # In upright position, gravity should point straight down in base frame
        expected = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        torch.testing.assert_close(gravity, expected)

    def test_get_base_projected_gravity_rotated_90x(self):
        """Test gravity projection when robot is rotated 90 degrees around x-axis"""
        # Create quaternion for 90-degree rotation around x-axis
        quat = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
        # Convert to scalar-first format [w, x, y, z]
        self.robot._h1_sdk.torso_orientation = quat

        # Call the function
        gravity = self.robot.get_base_projected_gravity()

        # After 90-degree x rotation, gravity should point along negative y in base frame
        expected = torch.tensor([[0.0, -1.0, 0.0]], device=self.device)
        torch.testing.assert_close(gravity, expected, atol=1e-6, rtol=1e-6)

    def test_get_base_projected_gravity_rotated_45y(self):
        """Test gravity projection when robot is rotated 45 degrees around y-axis"""
        # Create quaternion for 45-degree rotation around y-axis
        quat = R.from_euler("y", 45, degrees=True).as_quat(scalar_first=True)
        # Convert to scalar-first format [w, x, y, z]
        self.robot._h1_sdk.torso_orientation = quat

        # Call the function
        gravity = self.robot.get_base_projected_gravity()

        # After 45-degree y rotation, gravity should be split between x and z
        expected = torch.tensor([[0.707107, 0.0, -0.707107]], device=self.device)
        torch.testing.assert_close(gravity, expected, atol=1e-6, rtol=1e-6)

    def test_get_base_projected_gravity_inverted(self):
        """Test gravity projection when robot is completely inverted"""
        # Create quaternion for 180-degree rotation around x-axis
        quat = R.from_euler("x", 180, degrees=True).as_quat(scalar_first=True)
        self.robot._h1_sdk.torso_orientation = quat

        # Call the function
        gravity = self.robot.get_base_projected_gravity()

        # When inverted, gravity should point upward in base frame
        expected = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        torch.testing.assert_close(gravity, expected, atol=1e-6, rtol=1e-6)

    def test_get_base_projected_gravity_rotated_90z(self):
        """Test gravity projection when robot is rotated 90 degrees around z-axis"""
        # Create quaternion for 90-degree rotation around z-axis
        quat = R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
        self.robot._h1_sdk.torso_orientation = quat

        # Call the function
        gravity = self.robot.get_base_projected_gravity()

        # After 90-degree z rotation, gravity should still point straight down
        # because z-axis rotation doesn't affect gravity direction
        expected = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        torch.testing.assert_close(gravity, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
