# # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from __future__ import annotations

# import json
# import numpy as np
# import time
# import torch
# from dataclasses import dataclass, field
# from pathlib import Path
# from tqdm import tqdm

# from phc.smpllib.smpl_eval import compute_metrics_lite

# from neural_wbc.core import EnvironmentWrapper, math_utils


# @dataclass
# class Frame:
#     """A frame contains a set of trajectory data."""

#     body_pos: torch.Tensor = torch.tensor([])  # [num_links, 3]
#     body_pos_masked: torch.Tensor | None = None  # [num_links, 3]
#     upper_body_joint_pos: torch.Tensor = torch.tensor([])  # [num_joints, 3]
#     lower_body_joint_pos: torch.Tensor = torch.tensor([])  # [num_joints, 3]
#     root_pos: torch.Tensor = torch.tensor([])  # [3]
#     root_lin_vel: torch.Tensor = torch.tensor([])  # [3]
#     root_rot: torch.Tensor = torch.tensor([])  # [4]

#     @staticmethod
#     def from_dict(data: dict) -> Frame:
#         """Create a Frame from a dictionary."""
#         return Frame(**data)


# @dataclass
# class Episode:
#     """An episode contains a set of trajectories from different environments, each containing a sequence of frame data.

#     Each environment's trajectory has a maximum number of frames specified by max_frames_per_env. The episode stores
#     various motion attributes like body positions, joint positions, root positions/velocities etc. for each frame
#     in each environment's trajectory.
#     """

#     max_frames_per_env: torch.Tensor
#     body_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_links, 3)]
#     body_pos_masked: list[torch.Tensor] | None = None  # [num_envs, (num_frames, num_links, 3)]
#     upper_body_joint_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_joints, 3)]
#     lower_body_joint_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_joints, 3)]
#     root_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 3)]
#     root_lin_vel: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 3)]
#     root_rot: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 4)]

#     def __post_init__(self):
#         # Counter for number of frames added to the episode so far
#         self._num_frames_added = 0

#         # Dictionary to temporarily store frame data before aggregating into episode
#         # Keys are frame attributes (e.g. 'body_pos', 'root_pos')
#         # Values are tensors/arrays holding data for all frames and environments
#         self._frames = {}

#     def _init_empty_frames(self, frame: Frame):
#         """Initialize empty frame buffers to store trajectory data for all environments.

#         Creates zero-filled tensors/arrays sized to hold the maximum possible number of frames
#         and environments, matching the data types and shapes of the input frame.
#         """
#         max_possible_frames = max(self.max_frames_per_env).item()
#         max_possible_envs = len(self.max_frames_per_env)
#         for attr in vars(frame):
#             frame_data = getattr(frame, attr)
#             if frame_data is not None:
#                 data_shape = frame_data.shape
#                 if isinstance(frame_data, torch.Tensor):
#                     self._frames[attr] = torch.zeros(
#                         (max_possible_frames, max_possible_envs, *data_shape[1:]), device=frame_data.device
#                     )

#     @property
#     def num_envs(self) -> int:
#         """Number of environments in the episode."""
#         return len(self.body_pos)

#     def add_frame(self, frame: Frame):
#         """Add a frame to each trajectory in the episode.

#         Args:
#             frame (Frame): Frame containing trajectory data for all environments at this timestep
#         """
#         # Initialize frame buffers if this is the first frame being added
#         if len(self._frames) == 0:
#             self._init_empty_frames(frame)

#         # Store each frame attribute (body positions, joint angles etc) in the corresponding buffer
#         for attr in vars(frame):
#             # Get the data for this attribute from the frame
#             frame_data = getattr(frame, attr)

#             # Only store tensor data, skip other attributes
#             if isinstance(frame_data, torch.Tensor):
#                 # Add the frame data at the current timestep index
#                 self._frames[attr][self._num_frames_added] = frame_data

#         # Increment counter tracking number of frames added
#         self._num_frames_added += 1

#     def complete(self):
#         """Aggregate frames into episode data more efficiently.

#         Instead of splitting data environment by environment, we can use tensor operations
#         to split all environments at once, significantly reducing loop overhead.
#         """
#         num_envs = self.max_frames_per_env.shape[0]
#         for key, data in self._frames.items():
#             if isinstance(data, torch.Tensor):
#                 setattr(self, key, [data[:, i] for i in range(num_envs)])

#     def filter(self, ids: list[int]) -> Episode:
#         """Filter episode data to only include specified environment indices."""
#         # Create new empty episode to store filtered data
#         filtered = Episode(self.max_frames_per_env)

#         # Iterate through all attributes of this episode
#         for attr, data in vars(self).items():
#             # Only process attributes that are lists
#             if isinstance(data, list):
#                 # For each attribute, keep only the trajectories at the specified indices
#                 setattr(filtered, attr, [data[i] for i in ids])
#         return filtered

#     def trim(self, terminated_frame: torch.Tensor, end_id: int):
#         """Helper method to cut data based on terminated frame.

#         This function creates a new Episode object with truncated data. For each environment,
#         it keeps only the frames up to the termination point specified in terminated_frame.
#         It then further filters to keep only environments up to end_id.

#         Args:
#             terminated_frame: Tensor containing the frame index where each env terminated
#             end_id: Only keep environments up to this index

#         Returns:
#             A new Episode object with the trimmed data
#         """
#         trimmed = Episode(self.max_frames_per_env)
#         for attr, data in vars(self).items():
#             if isinstance(data, list):
#                 setattr(trimmed, attr, [data[i][: terminated_frame[i]] for i in range(len(data))][:end_id])
#         return trimmed


# class MotionTrackingMetrics:
#     """Class that aggregates motion tracking metrics throughout an evaluation."""

#     def __init__(self):
#         self.num_motions = 0
#         self.success_rate = 0.0
#         self.all_metrics = {}
#         self.success_metrics = {}
#         self.all_metrics_masked = {}
#         self.success_metrics_masked = {}

#         # Episodic data are stored as {"metric_name": {"means": [...], "weights": [...]}}
#         # The final metrics are weighted sums of the means.
#         self._all_metrics_by_episode = {}
#         self._all_metrics_masked_by_episode = {}
#         self._success_metrics_by_episode = {}
#         self._success_metrics_masked_by_episode = {}

#     def update(
#         self,
#         episode: Episode,
#         episode_gt: Episode,
#         success_ids: list,
#     ):
#         """Update and compute metrics for trajectories from all simulation instances in one episode."""
#         self.num_motions += episode.num_envs
#         # First, compute metrics on trajectories from all instances.
#         self._compute_metrics(episode, episode_gt, self._all_metrics_by_episode, self._all_metrics_masked_by_episode)

#         if len(success_ids) == 0:
#             return

#         # Then, collect the trajectory from successful instances and compute metrics.
#         success_episodes = episode.filter(success_ids)
#         success_episodes_gt = episode_gt.filter(success_ids)
#         self._compute_metrics(
#             success_episodes,
#             success_episodes_gt,
#             self._success_metrics_by_episode,
#             self._success_metrics_masked_by_episode,
#         )

#     def _compute_metrics(
#         self,
#         episode: Episode,
#         episode_gt: Episode,
#         storage: dict[str, dict[str, list[float]]],
#         masked_storage: dict[str, dict[str, list[float]]],
#     ):
#         # Get number of frames for each environment in the episode
#         num_frames = [body_pos.shape[0] for body_pos in episode.body_pos]
#         # Create weights for each frame that sum to 1.0 within each environment's trajectory
#         # This ensures each environment contributes equally to metrics regardless of trajectory length
#         # Shape: (sum(num_frames),)
#         frame_weights = torch.cat([torch.ones(n, device=episode.root_pos[0].device) / n for n in num_frames])

#         self._compute_link_metrics(
#             body_pos=episode.body_pos,
#             body_pos_gt=episode_gt.body_pos,
#             storage=storage,
#         )
#         self._compute_joint_metrics(
#             episode=episode,
#             episode_gt=episode_gt,
#             storage=storage,
#             frame_weights=frame_weights,
#         )
#         self._compute_root_metrics(
#             episode=episode,
#             episode_gt=episode_gt,
#             storage=storage,
#             frame_weights=frame_weights,
#         )
#         # if episode.body_pos_masked and episode_gt.body_pos_masked:
#         #     self._compute_link_metrics(
#         #         body_pos=episode.body_pos_masked,
#         #         body_pos_gt=episode_gt.body_pos_masked,
#         #         storage=masked_storage,
#         #     )

#     def _compute_link_metrics(
#         self,
#         body_pos: list[torch.Tensor],
#         body_pos_gt: list[torch.Tensor],
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
#         # compute_metrics_lite expects list of numpy arrays
#         body_pos_np = [body_pos_i.cpu().numpy() for body_pos_i in body_pos]
#         body_pos_gt_np = [body_pos_gt_i.cpu().numpy() for body_pos_gt_i in body_pos_gt]
#         metrics = compute_metrics_lite(body_pos_np, body_pos_gt_np)
#         for key, value in metrics.items():
#             self._record_metrics(key, np.mean(value).item(), value.size, storage)

#     def _compute_joint_metrics(
#         self,
#         episode: Episode,
#         episode_gt: Episode,
#         frame_weights: torch.Tensor,
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
#         self._compute_joint_tracking_error(
#             episode.upper_body_joint_pos,
#             episode_gt.upper_body_joint_pos,
#             frame_weights,
#             episode.num_envs,
#             storage,
#             "upper_body_joints_dist",
#         )
#         self._compute_joint_tracking_error(
#             episode.lower_body_joint_pos,
#             episode_gt.lower_body_joint_pos,
#             frame_weights,
#             episode.num_envs,
#             storage,
#             "lower_body_joints_dist",
#         )

#     def _compute_joint_tracking_error(
#         self,
#         joint_pos: list[torch.Tensor],
#         joint_pos_gt: list[torch.Tensor],
#         frame_weights: torch.Tensor,
#         num_envs: int,
#         storage: dict[str, dict[str, list[float]]],
#         name: str,
#     ):
#         """Compute joint tracking error."""

#         @torch.jit.script
#         def compute_joint_tracking_error(
#             joint_pos: torch.Tensor, joint_pos_gt: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
#         ) -> float:
#             """Compute weighted mean absolute joint position error across environments.

#             For each environment:
#             1. Take absolute difference between predicted and ground truth joint positions
#             2. Weight the differences by frame_weights to normalize across varying trajectory lengths
#             3. Take mean across joints

#             Finally, sum across environments and divide by num_envs for mean error.
#             """
#             return torch.sum(torch.mean(torch.abs(joint_pos - joint_pos_gt), dim=1) * frame_weights).item() / num_envs

#         # Concatenate joint positions across environments for faster computation
#         joint_pos_cat = torch.cat(joint_pos, dim=0)  # Shape: (sum(num_frames), num_joints)
#         joint_pos_gt_cat = torch.cat(joint_pos_gt, dim=0)  # Shape: (sum(num_frames), num_joints)
#         joint_tracking_error = compute_joint_tracking_error(joint_pos_cat, joint_pos_gt_cat, frame_weights, num_envs)
#         self._record_metrics(name, joint_tracking_error, num_envs, storage)

#     def _compute_root_metrics(
#         self,
#         episode: Episode,
#         episode_gt: Episode,
#         frame_weights: torch.Tensor,
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute root metrics"""

#         # Concatenate root positions across environments for faster computation
#         root_pos_cat = torch.cat(episode.root_pos, dim=0)  # Shape: (sum(num_frames), 3)
#         root_pos_gt_cat = torch.cat(episode_gt.root_pos, dim=0)  # Shape: (sum(num_frames), 3)
#         root_vel_cat = torch.cat(episode.root_lin_vel, dim=0)  # Shape: (sum(num_frames), 3)
#         root_vel_gt_cat = torch.cat(episode_gt.root_lin_vel, dim=0)  # Shape: (sum(num_frames), 3)
#         root_rot_cat = torch.cat(episode.root_rot, dim=0)  # Shape: (sum(num_frames), 4)
#         root_rot_gt_cat = torch.cat(episode_gt.root_rot, dim=0)  # Shape: (sum(num_frames), 4)

#         self._compute_root_height_error(root_pos_cat, root_pos_gt_cat, frame_weights, episode.num_envs, storage)
#         self._compute_root_vel_tracking_error(
#             root_vel_cat, root_rot_cat, root_vel_gt_cat, root_rot_gt_cat, frame_weights, episode.num_envs, storage
#         )
#         self._compute_root_rot_tracking_error(root_rot_cat, root_rot_gt_cat, frame_weights, episode.num_envs, storage)

#     def _compute_root_height_error(
#         self,
#         root_pos: torch.Tensor,
#         root_pos_gt: torch.Tensor,
#         frame_weights: torch.Tensor,
#         num_envs: int,
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute root height error."""

#         @torch.jit.script
#         def compute_height_error(
#             root_pos: torch.Tensor, root_pos_gt: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
#         ) -> float:
#             """Compute weighted mean absolute height error across environments.

#             For each environment:
#             1. Takes absolute difference between predicted and ground truth root z-coordinates
#             2. Weights the differences by frame_weights to normalize across varying trajectory lengths
#             3. Takes mean across frames

#             Finally, sum across environments and divide by num_envs for mean error.
#             """
#             return torch.sum(torch.abs(root_pos[:, 2] - root_pos_gt[:, 2]) * frame_weights).item() / num_envs

#         root_height_error = compute_height_error(root_pos, root_pos_gt, frame_weights, num_envs)
#         self._record_metrics("root_height_error", root_height_error, num_envs, storage)

#     def _compute_root_vel_tracking_error(
#         self,
#         root_vel: torch.Tensor,
#         root_rot: torch.Tensor,
#         root_vel_gt: torch.Tensor,
#         root_rot_gt: torch.Tensor,
#         frame_weights: torch.Tensor,
#         num_envs: int,
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute root velocity tracking error."""

#         @torch.jit.script
#         def compute_vel_error(
#             vel: torch.Tensor,
#             rot: torch.Tensor,
#             vel_gt: torch.Tensor,
#             rot_gt: torch.Tensor,
#             frame_weights: torch.Tensor,
#             num_envs: int,
#         ) -> float:
#             """Compute weighted mean velocity tracking error across environments.

#             For each environment:
#             1. Convert velocities to local frame using inverse rotation
#             2. Take L2 norm of difference between predicted and ground truth local velocities
#             3. Weight by frame_weights and average across frames

#             Finally, sum across environments and divide by num_envs for mean error.
#             """
#             local_vel_gt = math_utils.quat_rotate_inverse(rot_gt, vel_gt)
#             local_vel = math_utils.quat_rotate_inverse(rot, vel)
#             return torch.sum(torch.linalg.norm(local_vel - local_vel_gt, dim=-1) * frame_weights).item() / num_envs

#         root_vel_tracking_error = compute_vel_error(
#             root_vel, root_rot, root_vel_gt, root_rot_gt, frame_weights, num_envs
#         )
#         self._record_metrics("root_vel_tracking_error", root_vel_tracking_error, num_envs, storage)

#     def _compute_root_rot_tracking_error(
#         self,
#         root_rot: torch.Tensor,
#         root_rot_gt: torch.Tensor,
#         frame_weights: torch.Tensor,
#         num_envs: int,
#         storage: dict[str, dict[str, list[float]]],
#     ):
#         """Compute root rotation tracking error.

#         Args:
#             root_rot: Root rotation quaternions
#             root_rot_gt: Ground truth root rotation quaternions

#         Returns:
#             dict: Dictionary containing roll, pitch, yaw errors
#         """

#         @torch.jit.script
#         def compute_rot_tracking_error(
#             quat1: torch.Tensor, quat2: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
#         ) -> tuple[float, float, float]:
#             """Compute weighted mean rotation tracking error across environments.

#             For each environment:
#             1. Compute quaternion difference between predicted and ground truth rotations
#             2. Convert difference quaternion to Euler angles (roll, pitch, yaw)
#             3. Take absolute value of angles and weight by frame_weights
#             4. Average across frames

#             Finally, sum across environments and divide by num_envs for mean error.

#             Returns:
#                 tuple[float, float, float]: Mean roll, pitch and yaw errors in radians
#             """
#             quat_diff = math_utils.quat_mul(quat1, math_utils.quat_conjugate(quat2))
#             roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_diff)
#             roll_error = torch.sum(torch.abs(roll) * frame_weights).item() / num_envs
#             pitch_error = torch.sum(torch.abs(pitch) * frame_weights).item() / num_envs
#             yaw_error = torch.sum(torch.abs(yaw) * frame_weights).item() / num_envs
#             return roll_error, pitch_error, yaw_error

#         roll_error, pitch_error, yaw_error = compute_rot_tracking_error(root_rot, root_rot_gt, frame_weights, num_envs)
#         self._record_metrics("root_r_error", roll_error, num_envs, storage)
#         self._record_metrics("root_p_error", pitch_error, num_envs, storage)
#         self._record_metrics("root_y_error", yaw_error, num_envs, storage)

#     def _record_metrics(self, name: str, mean: float, weight: int, storage: dict[str, dict[str, list[float]]]):
#         """Record metrics by their means and number of elements (as weights)."""
#         if name not in storage:
#             storage[name] = {"means": [], "weights": []}
#         storage[name]["means"].append(mean)
#         storage[name]["weights"].append(weight)

#     def conclude(self):
#         """At the end of the evaluation, computes the metrics over all tasks."""
#         self.all_metrics = {
#             key: np.average(value["means"], weights=value["weights"])
#             for key, value in self._all_metrics_by_episode.items()
#         }
#         self.success_metrics = {
#             key: np.average(value["means"], weights=value["weights"])
#             for key, value in self._success_metrics_by_episode.items()
#         }
#         self.all_metrics_masked = {
#             key: np.average(value["means"], weights=value["weights"])
#             for key, value in self._all_metrics_masked_by_episode.items()
#         }
#         self.success_metrics_masked = {
#             key: np.average(value["means"], weights=value["weights"])
#             for key, value in self._success_metrics_masked_by_episode.items()
#         }

#     def print(self):
#         """Prints current metrics."""
#         print(f"Number of reference motions: {self.num_motions}")
#         print(f"Success Rate: {self.success_rate:.10f}")
#         print("All: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics.items()]))
#         print("Succ: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics.items()]))
#         print("All Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics_masked.items()]))
#         print("Succ Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics_masked.items()]))

#     def save(self, directory: str):
#         """Saves metrics to a time-stamped json file in ``directory``.

#         Args:
#             directory (str): Directory to stored the file to.
#         """
#         file_dir = Path(directory)
#         file_dir.mkdir(parents=True, exist_ok=True)
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         file_name = f"{timestamp}.json"
#         file_dir.joinpath(file_name)

#         content = {
#             "num_motions": self.num_motions,
#             "success_rate": self.success_rate,
#             "all": self.all_metrics,
#             "success": self.success_metrics,
#             "all_masked": self.all_metrics_masked,
#             "success_masked": self.success_metrics_masked,
#         }

#         with open(file_dir.joinpath(file_name), "w") as fh:
#             json.dump(content, fh)


# class Evaluator:
#     """A class for collecting data to evaluate motion tracking performance of an RL policy."""

#     def __init__(
#         self,
#         env_wrapper: EnvironmentWrapper,
#         metrics_path: str | None = None,
#     ):
#         """Initializes the evaluator.

#         Args:
#             env_wrapper (EnvironmentWrapper): The environment that the evaluation is taking place.
#             metrics_path (str | None, optional): The directory that the metrics will be saved to. Defaults to None.
#         """
#         self._num_envs = env_wrapper.num_envs
#         self._device = env_wrapper.device
#         self._ref_motion_mgr = env_wrapper.reference_motion_manager

#         self._ref_motion_start_id = 0
#         self._num_unique_ref_motions = self._ref_motion_mgr.num_unique_motions
#         self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
#         self._metrics = MotionTrackingMetrics()
#         self._metrics_path = metrics_path

#         # Episode data
#         self._terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
#         self._terminated_frame = self._ref_motion_frames.detach().clone()
#         self._failed = torch.zeros((self._num_unique_ref_motions), dtype=torch.bool, device=self._device)
#         self._episode = Episode(max_frames_per_env=self._ref_motion_frames)
#         self._episode_gt = Episode(max_frames_per_env=self._ref_motion_frames)

#         # Status
#         self._pbar = tqdm(range(self._num_unique_ref_motions // self._num_envs), position=0, leave=True)
#         self._curr_steps = 0
#         self._num_episodes = 0

#     def collect(self, dones: torch.Tensor, info: dict) -> bool:
#         """Collects data from a step and updates internal states.

#         Args:
#             dones (torch.Tensor): environments that are terminated (failed) or truncated (timed out).
#             info (dict): Extra information collected from a step.

#         Returns:
#             bool: Whether all current reference motions are evaluated and that all environments need a reset.
#         """
#         self._curr_steps += 1
#         # Get the environments that terminated at the most recent step
#         newly_terminated = torch.logical_and(~self._terminated, dones)
#         self._collect_step_data(newly_terminated, info=info)

#         self._terminated_frame[newly_terminated] = self._curr_steps
#         self._terminated = torch.logical_or(self._terminated, dones)

#         update_str = self._update_status_bar()

#         if self._terminated.sum() == self._num_envs:
#             self._aggregate_data()
#             self._num_episodes += self._num_envs
#             print(update_str)
#             return True

#         return False

#     def _collect_step_data(self, newly_terminated: torch.Tensor, info: dict):
#         """Collects data after each step.

#         Args:
#             newly_terminated(torch.Tensor(bool)): Newly terminated env
#             info (dict): Extra information collected from a step.
#         """

#         state_data = info["data"]["state"]
#         ground_truth_data = info["data"]["ground_truth"]

#         body_pos = state_data["body_pos"]
#         num_envs, num_bodies, _ = body_pos.shape

#         mask = info["data"]["mask"]
#         body_mask = mask[:, :num_bodies]
#         body_mask = body_mask.unsqueeze(-1)
#         body_mask_expanded = body_mask.expand(num_envs, num_bodies, 3)

#         upper_body_joint_ids = info["data"]["upper_joint_ids"]
#         lower_body_joint_ids = info["data"]["lower_joint_ids"]

#         frame = self._build_frame(state_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids)
#         frame_gt = self._build_frame(
#             ground_truth_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids
#         )

#         self._update_failure_metrics(newly_terminated, info)
#         self._episode.add_frame(frame)
#         self._episode_gt.add_frame(frame_gt)

#     def _build_frame(
#         self, data: dict, mask: torch.Tensor, num_envs: int, upper_joint_ids: list, lower_joint_ids: list
#     ) -> Frame:
#         """Builds a frame from the data and mask.

#         Args:
#             data (dict): Dictionary containing trajectory data including body positions, joint positions, etc.
#             mask (torch.Tensor): Boolean mask array indicating which bodies to include in masked data.
#             num_envs (int): Number of environments.
#             upper_joint_ids (list): List of indices for upper body joints.
#             lower_joint_ids (list): List of indices for lower body joints.

#         Returns:
#             Frame: A Frame object containing the processed trajectory data.
#         """
#         # if torch.any(mask):
#         #     # breakpoint()
#         #     # mask: 10, 23, 3
#         #     # data["body_pos"]: 10, 23, 3
#         #     print(mask.shape)
#         #     print(data["body_pos"].shape)
#         #     print(data["body_pos"][mask].shape)
#         #     print("-----------------")

#         #     data["body_pos_masked"] = data["body_pos"][mask].reshape(num_envs, -1, 3)
#         # else:
#         #     data["body_pos_masked"] = None

#         joint_pos = data.pop("joint_pos")
#         data["upper_body_joint_pos"] = joint_pos[:, upper_joint_ids]
#         data["lower_body_joint_pos"] = joint_pos[:, lower_joint_ids]
#         return Frame.from_dict(data)

#     def _update_failure_metrics(self, newly_terminated: torch.Tensor, info: dict):
#         """Updates failure metrics based on termination conditions."""
#         start_id = self._ref_motion_start_id
#         end_id = min(self._ref_motion_start_id + self._num_envs, self._num_unique_ref_motions)
#         counted_envs = end_id - start_id

#         # Get failure conditions excluding reference motion length
#         failed_conditions = torch.stack(
#             [
#                 v[:counted_envs].flatten()
#                 for k, v in info["termination_conditions"].items()
#                 if k != "reference_motion_length"
#             ]
#         )

#         # Update failed environments
#         self._failed[start_id:end_id] |= torch.logical_and(
#             newly_terminated[:counted_envs], torch.any(failed_conditions, dim=0)
#         )
#         self._metrics.success_rate = 1 - torch.sum(self._failed).item() / end_id

#     def _aggregate_data(self):
#         """Aggregates data from one episode."""
#         motion_end_id = min(self._num_envs, self._num_unique_ref_motions - self._num_episodes)
#         self._episode.complete()
#         self._episode_gt.complete()
#         self._episode = self._episode.trim(self._terminated_frame, motion_end_id)
#         self._episode_gt = self._episode_gt.trim(self._terminated_frame, motion_end_id)

#         # Get success IDs
#         start_id = self._ref_motion_start_id
#         end_id = min(start_id + self._num_envs, self._num_unique_ref_motions)
#         success_ids = torch.nonzero(~self._failed[start_id:end_id]).flatten().tolist()

#         # Update metrics
#         self._metrics.update(
#             episode=self._episode,
#             episode_gt=self._episode_gt,
#             success_ids=success_ids,
#         )

#     def _reset_data_buffer(self):
#         """Resets data buffer for new episodes."""
#         self._terminated[:] = False
#         self._pbar.update(1)
#         self._pbar.refresh()
#         self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
#         self._episode = Episode(max_frames_per_env=self._ref_motion_frames)
#         self._episode_gt = Episode(max_frames_per_env=self._ref_motion_frames)
#         self._terminated_frame = self._ref_motion_frames.detach().clone()
#         self._curr_steps = 0

#     def _update_status_bar(self):
#         """Updates status bar in the console to display current progress and selected metrics."""
#         update_str = (
#             f"Terminated: {self._terminated.sum().item()}  | max frames: {self._ref_motion_frames.max()} | steps"
#             f" {self._curr_steps} | Start: {self._ref_motion_start_id} | Succ rate: {self._metrics.success_rate:.3f} |"
#             f" Total failures: {self._failed.sum().item()} "
#         )
#         self._pbar.set_description(update_str)
#         return update_str

#     def is_evaluation_complete(self) -> bool:
#         """Returns whether all reference motions in the dataset have been evaluated.

#         Returns:
#             bool: True if all reference motions have been evaluated.
#         """
#         return self._num_episodes >= self._num_unique_ref_motions

#     def conclude(self):
#         """Concludes evaluation by computing, printing and optionally saving metrics."""
#         self._pbar.close()
#         self._metrics.conclude()
#         self._metrics.print()
#         if self._metrics_path:
#             self._metrics.save(self._metrics_path)

#     def forward_motion_samples(self):
#         """Steps forward in the list of reference motions.

#         All simulated environments must be reset following this function call.
#         """
#         self._ref_motion_start_id += self._num_envs
#         self._ref_motion_mgr.load_motions(random_sample=False, start_idx=self._ref_motion_start_id)
#         self._reset_data_buffer()

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

import json
import numpy as np
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from phc.smpllib.smpl_eval import compute_metrics_lite

from neural_wbc.core import EnvironmentWrapper, math_utils


@dataclass
class Frame:
    """A frame contains a set of trajectory data."""

    body_pos: torch.Tensor = torch.tensor([])  # [num_links, 3]
    body_pos_masked: torch.Tensor | None = None  # [num_links, 3]
    upper_body_joint_pos: torch.Tensor = torch.tensor([])  # [num_joints, 3]
    lower_body_joint_pos: torch.Tensor = torch.tensor([])  # [num_joints, 3]
    root_pos: torch.Tensor = torch.tensor([])  # [3]
    root_lin_vel: torch.Tensor = torch.tensor([])  # [3]
    root_rot: torch.Tensor = torch.tensor([])  # [4]

    @staticmethod
    def from_dict(data: dict) -> Frame:
        """Create a Frame from a dictionary."""
        return Frame(**data)


@dataclass
class Episode:
    """An episode contains a set of trajectories from different environments, each containing a sequence of frame data.

    Each environment's trajectory has a maximum number of frames specified by max_frames_per_env. The episode stores
    various motion attributes like body positions, joint positions, root positions/velocities etc. for each frame
    in each environment's trajectory.
    """

    max_frames_per_env: torch.Tensor
    body_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_links, 3)]
    body_pos_masked: list[torch.Tensor] | None = None  # [num_envs, (num_frames, num_links, 3)]
    upper_body_joint_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_joints, 3)]
    lower_body_joint_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, num_joints, 3)]
    root_pos: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 3)]
    root_lin_vel: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 3)]
    root_rot: list[torch.Tensor] = field(default_factory=list)  # [num_envs, (num_frames, 4)]

    def __post_init__(self):
        # Counter for number of frames added to the episode so far
        self._num_frames_added = 0

        # Dictionary to temporarily store frame data before aggregating into episode
        # Keys are frame attributes (e.g. 'body_pos', 'root_pos')
        # Values are tensors/arrays holding data for all frames and environments
        self._frames = {}

    def _init_empty_frames(self, frame: Frame):
        """Initialize empty frame buffers to store trajectory data for all environments.

        Creates zero-filled tensors/arrays sized to hold the maximum possible number of frames
        and environments, matching the data types and shapes of the input frame.
        """
        max_possible_frames = max(self.max_frames_per_env).item()
        max_possible_envs = len(self.max_frames_per_env)
        for attr in vars(frame):
            frame_data = getattr(frame, attr)
            data_shape = frame_data.shape
            if isinstance(frame_data, torch.Tensor):
                self._frames[attr] = torch.zeros(
                    (max_possible_frames, max_possible_envs, *data_shape[1:]), device=frame_data.device
                )

    @property
    def num_envs(self) -> int:
        """Number of environments in the episode."""
        return len(self.body_pos)

    def add_frame(self, frame: Frame):
        """Add a frame to each trajectory in the episode.

        Args:
            frame (Frame): Frame containing trajectory data for all environments at this timestep
        """
        # Initialize frame buffers if this is the first frame being added
        if len(self._frames) == 0:
            self._init_empty_frames(frame)

        # Store each frame attribute (body positions, joint angles etc) in the corresponding buffer
        for attr in vars(frame):
            # Get the data for this attribute from the frame
            frame_data = getattr(frame, attr)

            # Only store tensor data, skip other attributes
            if isinstance(frame_data, torch.Tensor):
                # Add the frame data at the current timestep index
                self._frames[attr][self._num_frames_added] = frame_data

        # Increment counter tracking number of frames added
        self._num_frames_added += 1

    def complete(self):
        """Aggregate frames into episode data more efficiently.

        Instead of splitting data environment by environment, we can use tensor operations
        to split all environments at once, significantly reducing loop overhead.
        """
        num_envs = self.max_frames_per_env.shape[0]
        for key, data in self._frames.items():
            if isinstance(data, torch.Tensor):
                setattr(self, key, [data[:, i] for i in range(num_envs)])

    def filter(self, ids: list[int]) -> Episode:
        """Filter episode data to only include specified environment indices."""
        # Create new empty episode to store filtered data
        filtered = Episode(self.max_frames_per_env)

        # Iterate through all attributes of this episode
        for attr, data in vars(self).items():
            # Only process attributes that are lists
            if isinstance(data, list):
                # For each attribute, keep only the trajectories at the specified indices
                setattr(filtered, attr, [data[i] for i in ids])
        return filtered

    def trim(self, terminated_frame: torch.Tensor, end_id: int):
        """Helper method to cut data based on terminated frame.

        This function creates a new Episode object with truncated data. For each environment,
        it keeps only the frames up to the termination point specified in terminated_frame.
        It then further filters to keep only environments up to end_id.

        Args:
            terminated_frame: Tensor containing the frame index where each env terminated
            end_id: Only keep environments up to this index

        Returns:
            A new Episode object with the trimmed data
        """
        trimmed = Episode(self.max_frames_per_env)
        for attr, data in vars(self).items():
            if isinstance(data, list):
                setattr(trimmed, attr, [data[i][: terminated_frame[i]] for i in range(len(data))][:end_id])
        return trimmed


class MotionTrackingMetrics:
    """Class that aggregates motion tracking metrics throughout an evaluation."""

    def __init__(self):
        self.num_motions = 0
        self.success_rate = 0.0
        self.all_metrics = {}
        self.success_metrics = {}
        self.all_metrics_masked = {}
        self.success_metrics_masked = {}

        # Episodic data are stored as {"metric_name": {"means": [...], "weights": [...]}}
        # The final metrics are weighted sums of the means.
        self._all_metrics_by_episode = {}
        self._all_metrics_masked_by_episode = {}
        self._success_metrics_by_episode = {}
        self._success_metrics_masked_by_episode = {}

    def update(
        self,
        episode: Episode,
        episode_gt: Episode,
        success_ids: list,
    ):
        """Update and compute metrics for trajectories from all simulation instances in one episode."""
        self.num_motions += episode.num_envs
        # First, compute metrics on trajectories from all instances.
        self._compute_metrics(episode, episode_gt, self._all_metrics_by_episode, self._all_metrics_masked_by_episode)

        if len(success_ids) == 0:
            return

        # Then, collect the trajectory from successful instances and compute metrics.
        success_episodes = episode.filter(success_ids)
        success_episodes_gt = episode_gt.filter(success_ids)
        self._compute_metrics(
            success_episodes,
            success_episodes_gt,
            self._success_metrics_by_episode,
            self._success_metrics_masked_by_episode,
        )

    def _compute_metrics(
        self,
        episode: Episode,
        episode_gt: Episode,
        storage: dict[str, dict[str, list[float]]],
        masked_storage: dict[str, dict[str, list[float]]],
    ):
        # Get number of frames for each environment in the episode
        num_frames = [body_pos.shape[0] for body_pos in episode.body_pos]
        # Create weights for each frame that sum to 1.0 within each environment's trajectory
        # This ensures each environment contributes equally to metrics regardless of trajectory length
        # Shape: (sum(num_frames),)
        frame_weights = torch.cat([torch.ones(n, device=episode.root_pos[0].device) / n for n in num_frames])

        self._compute_link_metrics(
            body_pos=episode.body_pos,
            body_pos_gt=episode_gt.body_pos,
            storage=storage,
        )
        self._compute_joint_metrics(
            episode=episode,
            episode_gt=episode_gt,
            storage=storage,
            frame_weights=frame_weights,
        )
        self._compute_root_metrics(
            episode=episode,
            episode_gt=episode_gt,
            storage=storage,
            frame_weights=frame_weights,
        )
        if episode.body_pos_masked and episode_gt.body_pos_masked:
            self._compute_link_metrics(
                body_pos=episode.body_pos_masked,
                body_pos_gt=episode_gt.body_pos_masked,
                storage=masked_storage,
            )

    def _compute_link_metrics(
        self,
        body_pos: list[torch.Tensor],
        body_pos_gt: list[torch.Tensor],
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
        # compute_metrics_lite expects list of numpy arrays
        body_pos_np = [body_pos_i.cpu().numpy() for body_pos_i in body_pos]
        body_pos_gt_np = [body_pos_gt_i.cpu().numpy() for body_pos_gt_i in body_pos_gt]
        metrics = compute_metrics_lite(body_pos_np, body_pos_gt_np)
        for key, value in metrics.items():
            self._record_metrics(key, np.mean(value).item(), value.size, storage)

    def _compute_joint_metrics(
        self,
        episode: Episode,
        episode_gt: Episode,
        frame_weights: torch.Tensor,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute metrics of trajectories and save them by their means and number of elements (as weights)."""
        self._compute_joint_tracking_error(
            episode.upper_body_joint_pos,
            episode_gt.upper_body_joint_pos,
            frame_weights,
            episode.num_envs,
            storage,
            "upper_body_joints_dist",
        )
        self._compute_joint_tracking_error(
            episode.lower_body_joint_pos,
            episode_gt.lower_body_joint_pos,
            frame_weights,
            episode.num_envs,
            storage,
            "lower_body_joints_dist",
        )

    def _compute_joint_tracking_error(
        self,
        joint_pos: list[torch.Tensor],
        joint_pos_gt: list[torch.Tensor],
        frame_weights: torch.Tensor,
        num_envs: int,
        storage: dict[str, dict[str, list[float]]],
        name: str,
    ):
        """Compute joint tracking error."""

        @torch.jit.script
        def compute_joint_tracking_error(
            joint_pos: torch.Tensor, joint_pos_gt: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
        ) -> float:
            """Compute weighted mean absolute joint position error across environments.

            For each environment:
            1. Take absolute difference between predicted and ground truth joint positions
            2. Weight the differences by frame_weights to normalize across varying trajectory lengths
            3. Take mean across joints

            Finally, sum across environments and divide by num_envs for mean error.
            """
            return torch.sum(torch.mean(torch.abs(joint_pos - joint_pos_gt), dim=1) * frame_weights).item() / num_envs

        # Concatenate joint positions across environments for faster computation
        joint_pos_cat = torch.cat(joint_pos, dim=0)  # Shape: (sum(num_frames), num_joints)
        joint_pos_gt_cat = torch.cat(joint_pos_gt, dim=0)  # Shape: (sum(num_frames), num_joints)
        joint_tracking_error = compute_joint_tracking_error(joint_pos_cat, joint_pos_gt_cat, frame_weights, num_envs)
        self._record_metrics(name, joint_tracking_error, num_envs, storage)

    def _compute_root_metrics(
        self,
        episode: Episode,
        episode_gt: Episode,
        frame_weights: torch.Tensor,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute root metrics"""

        # Concatenate root positions across environments for faster computation
        root_pos_cat = torch.cat(episode.root_pos, dim=0)  # Shape: (sum(num_frames), 3)
        root_pos_gt_cat = torch.cat(episode_gt.root_pos, dim=0)  # Shape: (sum(num_frames), 3)
        root_vel_cat = torch.cat(episode.root_lin_vel, dim=0)  # Shape: (sum(num_frames), 3)
        root_vel_gt_cat = torch.cat(episode_gt.root_lin_vel, dim=0)  # Shape: (sum(num_frames), 3)
        root_rot_cat = torch.cat(episode.root_rot, dim=0)  # Shape: (sum(num_frames), 4)
        root_rot_gt_cat = torch.cat(episode_gt.root_rot, dim=0)  # Shape: (sum(num_frames), 4)

        self._compute_root_height_error(root_pos_cat, root_pos_gt_cat, frame_weights, episode.num_envs, storage)
        self._compute_root_vel_tracking_error(
            root_vel_cat, root_rot_cat, root_vel_gt_cat, root_rot_gt_cat, frame_weights, episode.num_envs, storage
        )
        self._compute_root_rot_tracking_error(root_rot_cat, root_rot_gt_cat, frame_weights, episode.num_envs, storage)

    def _compute_root_height_error(
        self,
        root_pos: torch.Tensor,
        root_pos_gt: torch.Tensor,
        frame_weights: torch.Tensor,
        num_envs: int,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute root height error."""

        @torch.jit.script
        def compute_height_error(
            root_pos: torch.Tensor, root_pos_gt: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
        ) -> float:
            """Compute weighted mean absolute height error across environments.

            For each environment:
            1. Takes absolute difference between predicted and ground truth root z-coordinates
            2. Weights the differences by frame_weights to normalize across varying trajectory lengths
            3. Takes mean across frames

            Finally, sum across environments and divide by num_envs for mean error.
            """
            return torch.sum(torch.abs(root_pos[:, 2] - root_pos_gt[:, 2]) * frame_weights).item() / num_envs

        root_height_error = compute_height_error(root_pos, root_pos_gt, frame_weights, num_envs)
        self._record_metrics("root_height_error", root_height_error, num_envs, storage)

    def _compute_root_vel_tracking_error(
        self,
        root_vel: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel_gt: torch.Tensor,
        root_rot_gt: torch.Tensor,
        frame_weights: torch.Tensor,
        num_envs: int,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute root velocity tracking error."""

        @torch.jit.script
        def compute_vel_error(
            vel: torch.Tensor,
            rot: torch.Tensor,
            vel_gt: torch.Tensor,
            rot_gt: torch.Tensor,
            frame_weights: torch.Tensor,
            num_envs: int,
        ) -> float:
            """Compute weighted mean velocity tracking error across environments.

            For each environment:
            1. Convert velocities to local frame using inverse rotation
            2. Take L2 norm of difference between predicted and ground truth local velocities
            3. Weight by frame_weights and average across frames

            Finally, sum across environments and divide by num_envs for mean error.
            """
            local_vel_gt = math_utils.quat_rotate_inverse(rot_gt, vel_gt)
            local_vel = math_utils.quat_rotate_inverse(rot, vel)
            return torch.sum(torch.linalg.norm(local_vel - local_vel_gt, dim=-1) * frame_weights).item() / num_envs

        root_vel_tracking_error = compute_vel_error(
            root_vel, root_rot, root_vel_gt, root_rot_gt, frame_weights, num_envs
        )
        self._record_metrics("root_vel_tracking_error", root_vel_tracking_error, num_envs, storage)

    def _compute_root_rot_tracking_error(
        self,
        root_rot: torch.Tensor,
        root_rot_gt: torch.Tensor,
        frame_weights: torch.Tensor,
        num_envs: int,
        storage: dict[str, dict[str, list[float]]],
    ):
        """Compute root rotation tracking error.

        Args:
            root_rot: Root rotation quaternions
            root_rot_gt: Ground truth root rotation quaternions

        Returns:
            dict: Dictionary containing roll, pitch, yaw errors
        """

        @torch.jit.script
        def compute_rot_tracking_error(
            quat1: torch.Tensor, quat2: torch.Tensor, frame_weights: torch.Tensor, num_envs: int
        ) -> tuple[float, float, float]:
            """Compute weighted mean rotation tracking error across environments.

            For each environment:
            1. Compute quaternion difference between predicted and ground truth rotations
            2. Convert difference quaternion to Euler angles (roll, pitch, yaw)
            3. Take absolute value of angles and weight by frame_weights
            4. Average across frames

            Finally, sum across environments and divide by num_envs for mean error.

            Returns:
                tuple[float, float, float]: Mean roll, pitch and yaw errors in radians
            """
            quat_diff = math_utils.quat_mul(quat1, math_utils.quat_conjugate(quat2))
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_diff)
            roll_error = torch.sum(torch.abs(roll) * frame_weights).item() / num_envs
            pitch_error = torch.sum(torch.abs(pitch) * frame_weights).item() / num_envs
            yaw_error = torch.sum(torch.abs(yaw) * frame_weights).item() / num_envs
            return roll_error, pitch_error, yaw_error

        roll_error, pitch_error, yaw_error = compute_rot_tracking_error(root_rot, root_rot_gt, frame_weights, num_envs)
        self._record_metrics("root_r_error", roll_error, num_envs, storage)
        self._record_metrics("root_p_error", pitch_error, num_envs, storage)
        self._record_metrics("root_y_error", yaw_error, num_envs, storage)

    def _record_metrics(self, name: str, mean: float, weight: int, storage: dict[str, dict[str, list[float]]]):
        """Record metrics by their means and number of elements (as weights)."""
        if name not in storage:
            storage[name] = {"means": [], "weights": []}
        storage[name]["means"].append(mean)
        storage[name]["weights"].append(weight)

    def conclude(self):
        """At the end of the evaluation, computes the metrics over all tasks."""
        self.all_metrics = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._all_metrics_by_episode.items()
        }
        self.success_metrics = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._success_metrics_by_episode.items()
        }
        self.all_metrics_masked = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._all_metrics_masked_by_episode.items()
        }
        self.success_metrics_masked = {
            key: np.average(value["means"], weights=value["weights"])
            for key, value in self._success_metrics_masked_by_episode.items()
        }

    def print(self):
        """Prints current metrics."""
        print(f"Number of reference motions: {self.num_motions}")
        print(f"Success Rate: {self.success_rate:.10f}")
        print("All: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics.items()]))
        print("Succ: ", " \t\t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics.items()]))
        print("All Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.all_metrics_masked.items()]))
        print("Succ Masked: ", " \t".join([f"{k}: {v:.3f}" for k, v in self.success_metrics_masked.items()]))

    def save(self, directory: str):
        """Saves metrics to a time-stamped json file in ``directory``.

        Args:
            directory (str): Directory to stored the file to.
        """
        file_dir = Path(directory)
        file_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{timestamp}.json"
        file_dir.joinpath(file_name)

        content = {
            "num_motions": self.num_motions,
            "success_rate": self.success_rate,
            "all": self.all_metrics,
            "success": self.success_metrics,
            "all_masked": self.all_metrics_masked,
            "success_masked": self.success_metrics_masked,
        }

        with open(file_dir.joinpath(file_name), "w") as fh:
            json.dump(content, fh)


class Evaluator:
    """A class for collecting data to evaluate motion tracking performance of an RL policy."""

    def __init__(
        self,
        env_wrapper: EnvironmentWrapper,
        metrics_path: str | None = None,
    ):
        """Initializes the evaluator.

        Args:
            env_wrapper (EnvironmentWrapper): The environment that the evaluation is taking place.
            metrics_path (str | None, optional): The directory that the metrics will be saved to. Defaults to None.
        """
        self._num_envs = env_wrapper.num_envs
        self._device = env_wrapper.device
        self._ref_motion_mgr = env_wrapper.reference_motion_manager

        self._ref_motion_start_id = 0
        self._num_unique_ref_motions = self._ref_motion_mgr.num_unique_motions
        self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
        self._metrics = MotionTrackingMetrics()
        self._metrics_path = metrics_path

        # Episode data
        self._terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._terminated_frame = self._ref_motion_frames.detach().clone()
        self._failed = torch.zeros((self._num_unique_ref_motions), dtype=torch.bool, device=self._device)
        self._episode = Episode(max_frames_per_env=self._ref_motion_frames)
        self._episode_gt = Episode(max_frames_per_env=self._ref_motion_frames)

        # Status
        self._pbar = tqdm(range(self._num_unique_ref_motions // self._num_envs), position=0, leave=True)
        self._curr_steps = 0
        self._num_episodes = 0

    def collect(self, dones: torch.Tensor, info: dict) -> bool:
        """Collects data from a step and updates internal states.

        Args:
            dones (torch.Tensor): environments that are terminated (failed) or truncated (timed out).
            info (dict): Extra information collected from a step.

        Returns:
            bool: Whether all current reference motions are evaluated and that all environments need a reset.
        """
        self._curr_steps += 1
        # Get the environments that terminated at the most recent step
        newly_terminated = torch.logical_and(~self._terminated, dones)
        self._collect_step_data(newly_terminated, info=info)

        self._terminated_frame[newly_terminated] = self._curr_steps
        self._terminated = torch.logical_or(self._terminated, dones)

        update_str = self._update_status_bar()

        if self._terminated.sum() == self._num_envs:
            self._aggregate_data()
            self._num_episodes += self._num_envs
            print(update_str)
            return True

        return False

    def _collect_step_data(self, newly_terminated: torch.Tensor, info: dict):
        """Collects data after each step.

        Args:
            newly_terminated(torch.Tensor(bool)): Newly terminated env
            info (dict): Extra information collected from a step.
        """

        state_data = info["data"]["state"]
        ground_truth_data = info["data"]["ground_truth"]

        body_pos = state_data["body_pos"]
        num_envs, num_bodies, _ = body_pos.shape

        mask = info["data"]["mask"]
        body_mask = mask[:, :num_bodies]
        body_mask = body_mask.unsqueeze(-1)
        body_mask_expanded = body_mask.expand(num_envs, num_bodies, 3)

        upper_body_joint_ids = info["data"]["upper_joint_ids"]
        lower_body_joint_ids = info["data"]["lower_joint_ids"]

        frame = self._build_frame(state_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids)
        frame_gt = self._build_frame(
            ground_truth_data, body_mask_expanded, num_envs, upper_body_joint_ids, lower_body_joint_ids
        )

        self._update_failure_metrics(newly_terminated, info)
        self._episode.add_frame(frame)
        self._episode_gt.add_frame(frame_gt)

    def _build_frame(
        self, data: dict, mask: torch.Tensor, num_envs: int, upper_joint_ids: list, lower_joint_ids: list
    ) -> Frame:
        """Builds a frame from the data and mask.

        Args:
            data (dict): Dictionary containing trajectory data including body positions, joint positions, etc.
            mask (torch.Tensor): Boolean mask array indicating which bodies to include in masked data.
            num_envs (int): Number of environments.
            upper_joint_ids (list): List of indices for upper body joints.
            lower_joint_ids (list): List of indices for lower body joints.

        Returns:
            Frame: A Frame object containing the processed trajectory data.
        """
        if torch.any(mask):
            data["body_pos_masked"] = data["body_pos"][mask].reshape(num_envs, -1, 3)
        else:
            data["body_pos_masked"] = None

        joint_pos = data.pop("joint_pos")
        data["upper_body_joint_pos"] = joint_pos[:, upper_joint_ids]
        data["lower_body_joint_pos"] = joint_pos[:, lower_joint_ids]
        return Frame.from_dict(data)

    def _update_failure_metrics(self, newly_terminated: torch.Tensor, info: dict):
        """Updates failure metrics based on termination conditions."""
        start_id = self._ref_motion_start_id
        end_id = min(self._ref_motion_start_id + self._num_envs, self._num_unique_ref_motions)
        counted_envs = end_id - start_id

        # Get failure conditions excluding reference motion length
        failed_conditions = torch.stack(
            [
                v[:counted_envs].flatten()
                for k, v in info["termination_conditions"].items()
                if k != "reference_motion_length"
            ]
        )

        # Update failed environments
        self._failed[start_id:end_id] |= torch.logical_and(
            newly_terminated[:counted_envs], torch.any(failed_conditions, dim=0)
        )
        self._metrics.success_rate = 1 - torch.sum(self._failed).item() / end_id

    def _aggregate_data(self):
        """Aggregates data from one episode."""
        motion_end_id = min(self._num_envs, self._num_unique_ref_motions - self._num_episodes)
        self._episode.complete()
        self._episode_gt.complete()
        self._episode = self._episode.trim(self._terminated_frame, motion_end_id)
        self._episode_gt = self._episode_gt.trim(self._terminated_frame, motion_end_id)

        # Get success IDs
        start_id = self._ref_motion_start_id
        end_id = min(start_id + self._num_envs, self._num_unique_ref_motions)
        success_ids = torch.nonzero(~self._failed[start_id:end_id]).flatten().tolist()

        # Update metrics
        self._metrics.update(
            episode=self._episode,
            episode_gt=self._episode_gt,
            success_ids=success_ids,
        )

    def _reset_data_buffer(self):
        """Resets data buffer for new episodes."""
        self._terminated[:] = False
        self._pbar.update(1)
        self._pbar.refresh()
        self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps()
        self._episode = Episode(max_frames_per_env=self._ref_motion_frames)
        self._episode_gt = Episode(max_frames_per_env=self._ref_motion_frames)
        self._terminated_frame = self._ref_motion_frames.detach().clone()
        self._curr_steps = 0

    def _update_status_bar(self):
        """Updates status bar in the console to display current progress and selected metrics."""
        update_str = (
            f"Terminated: {self._terminated.sum().item()}  | max frames: {self._ref_motion_frames.max()} | steps"
            f" {self._curr_steps} | Start: {self._ref_motion_start_id} | Succ rate: {self._metrics.success_rate:.3f} |"
            f" Total failures: {self._failed.sum().item()} "
        )
        self._pbar.set_description(update_str)
        return update_str

    def is_evaluation_complete(self) -> bool:
        """Returns whether all reference motions in the dataset have been evaluated.

        Returns:
            bool: True if all reference motions have been evaluated.
        """
        return self._num_episodes >= self._num_unique_ref_motions

    def conclude(self):
        """Concludes evaluation by computing, printing and optionally saving metrics."""
        self._pbar.close()
        self._metrics.conclude()
        self._metrics.print()
        if self._metrics_path:
            self._metrics.save(self._metrics_path)

    def forward_motion_samples(self):
        """Steps forward in the list of reference motions.

        All simulated environments must be reset following this function call.
        """
        self._ref_motion_start_id += self._num_envs
        self._ref_motion_mgr.load_motions(random_sample=False, start_idx=self._ref_motion_start_id)
        self._reset_data_buffer()