# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0
import re
import numpy as np
import math

class Filter:
    def __init__(self, T1, Ts, num_dofs):
        self.T1 = T1
        self.Ts = Ts
        self.num_dofs = num_dofs

        self.V1 = 1 / Ts * (1 - math.exp(-Ts / T1))
        self.V2 = 1 / Ts * (1 - math.exp(-Ts / T1)) ** 2

        # Persistent state for DT1: shape [batch_size]
        self.x = [None] * num_dofs

    def dt1_filter(self, q):
        """
        q: numpy array of shape [num_dofs]
        Returns: estimated derivative (velocity), shape [num_dofs]
        """
        qp = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            if self.x[i] is None:
                self.x[i] = -1.0 / self.T1 * q[i]  # initialize state
                qp[i] = 0
            else:
                qp[i] = self.x[i] + self.V1 * q[i]
                self.x[i] = (math.exp(-self.Ts / self.T1) * self.x[i] +
                             (math.exp(-self.Ts / self.T1) - 1) * self.V1 * q[i])
        return qp

def assert_equal(lhs: any, rhs: any, name: str):
    """Assert that 2 values are equal and provide a useful error if not.

    Args:
        lhs: First value to compare
        rhs: Second value to compare
        name: Description of what is being compared, used in error messages
    """
    # Handle dictionary comparisons
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        _assert_dicts_equal(lhs, rhs, name)
    # Handle numeric comparisons
    elif isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        _assert_numbers_equal(lhs, rhs, name)
    # Handle all other types
    else:
        _assert_values_equal(lhs, rhs, name)


def _assert_dicts_equal(lhs: dict, rhs: dict, name: str):
    """Compare two dictionaries and raise assertion error with details if not equal."""
    lhs_keys = set(lhs.keys())
    rhs_keys = set(rhs.keys())

    # Check for missing keys
    only_in_lhs = lhs_keys - rhs_keys
    only_in_rhs = rhs_keys - lhs_keys

    # Check for value differences
    diff_values = _get_differing_values(lhs, rhs, lhs_keys & rhs_keys)

    # Build error message if there are any differences
    error_parts = []
    if only_in_lhs:
        error_parts.append(f"Keys only in first dict: {only_in_lhs}")
    if only_in_rhs:
        error_parts.append(f"Keys only in second dict: {only_in_rhs}")
    if diff_values:
        error_parts.append(f"Keys with different values: {diff_values}")

    if error_parts:
        raise AssertionError(f"{name}: Dictionaries are not equal:\n" + "\n".join(error_parts))


def _get_differing_values(lhs: dict, rhs: dict, common_keys: set) -> dict:
    """Compare values for common keys between two dicts, return dict of differences."""
    diff_values = {}
    for key in common_keys:
        if isinstance(lhs[key], (int, float)) and isinstance(rhs[key], (int, float)):
            if abs(lhs[key] - rhs[key]) >= 1e-6:
                diff_values[key] = (lhs[key], rhs[key])
        elif lhs[key] != rhs[key]:
            diff_values[key] = (lhs[key], rhs[key])
    return diff_values


def _assert_numbers_equal(lhs: float, rhs: float, name: str):
    """Assert that two numbers are equal within a small tolerance."""
    if abs(lhs - rhs) >= 1e-6:
        raise AssertionError(f"{name}: Values are not equal within tolerance: {lhs} != {rhs}")


def _assert_values_equal(lhs: any, rhs: any, name: str):
    """Assert that two non-numeric values are exactly equal."""
    if lhs != rhs:
        raise AssertionError(f"{name}: Values are not equal: {lhs} != {rhs}")


def get_matching_indices(patterns: list[str], values: list[str], allow_empty: bool = False) -> list[int]:
    """Get indices of all elements in values that match any of the regex patterns."""
    all_indices = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        indices = [i for i, v in enumerate(values) if regex.match(v)]
        if len(indices) == 0 and not allow_empty:
            raise ValueError(f"No matching indices found for pattern {pattern} in {values}")
        all_indices.update(indices)
    return list(all_indices)
