from dataclasses import dataclass
from tqdm import tqdm
import h5py

import numpy as np
import torch
from neural_wbc.core import EnvironmentWrapper
    
class Recorder:
    """Recorder class to record robot state information, reference motion id, and mask"""
    def __init__(self,
                 env_wrapper: EnvironmentWrapper,
                 log_path: str):
        """Initialize the recorder.
        Args:
            env_wrapper (EnvironmentWrapper): The environment wrapper to record from.
            log_path (str): The path to save the recorded data.
        """
        self._num_envs = env_wrapper.num_envs
        assert self._num_envs == 1, "Recorder only supports single env for now"
        self._device = env_wrapper.device
        self._ref_motion_mgr = env_wrapper.reference_motion_manager
        
        self._ref_motion_start_id = 0
        self._num_unique_ref_motions = self._ref_motion_mgr.num_unique_motions

        self.log_path = log_path

        # Status
        self._pbar = tqdm(range(self._num_unique_ref_motions // self._num_envs), position=0, leave=True)
        self._curr_steps = 0
        self._num_episodes = 0
        self._recorded_frames_total = []
        self.frames = None

    def initialize_frame(self, motion_id, max_steps: int):
        frame = {}
        frame["robot_state"] = {}
        frame["robot_state"]["root_pos"] = np.zeros((max_steps, 3), dtype=np.float32)
        frame["robot_state"]["root_lin_vel"] = np.zeros((max_steps, 3), dtype=np.float32)
        frame["robot_state"]["root_ang_vel"] = np.zeros((max_steps, 3), dtype=np.float32)
        frame["robot_state"]["root_rot"] = np.zeros((max_steps, 4), dtype=np.float32)
        frame["robot_state"]["joint_pos"] = np.zeros((max_steps, 23), dtype=np.float32)
        frame["robot_state"]["joint_vel"] = np.zeros((max_steps, 23), dtype=np.float32)
        frame["action"] = np.zeros((max_steps, 23), dtype=np.float32)
        frame["ref_motion_id"] = motion_id
        frame["mask"] = None
        frame["step"] = 0
        return frame


    def collect(self, dones: torch.Tensor, info: dict) -> bool:
        """Collect data from a step and updates internal states.
        
        Args:
            dones (torch.Tensor): A tensor indicating if the episode is done.
            info (dict): A dictionary containing additional information.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        if self.frames is None:
            self._ref_motion_frames = self._ref_motion_mgr.get_motion_num_steps().item()
            self.frames = self.initialize_frame(self._ref_motion_mgr.motion_lib._curr_motion_ids.item(), self._ref_motion_frames)
        if self.frames["mask"] is None:
            self.frames["mask"] = info["data"]["mask"].cpu().numpy()
        self.frames["robot_state"]["root_pos"][self._curr_steps, :] = info["data"]["state"]["root_pos"].cpu().numpy()
        self.frames["robot_state"]["root_lin_vel"][self._curr_steps, :] = info["data"]["state"]["root_lin_vel"].cpu().numpy()
        self.frames["robot_state"]["root_ang_vel"][self._curr_steps, :] = info["data"]["state"]["root_ang_vel"].cpu().numpy()
        self.frames["robot_state"]["root_rot"][self._curr_steps, :] = info["data"]["state"]["root_rot"].cpu().numpy()
        self.frames["robot_state"]["joint_pos"][self._curr_steps, :] = info["data"]["state"]["joint_pos"].cpu().numpy()
        self.frames["robot_state"]["joint_vel"][self._curr_steps, :] = info["data"]["state"]["joint_vel"].cpu().numpy()      
        self.frames["action"][self._curr_steps, :] = info["data"]["action"].cpu().numpy()
        self._curr_steps += 1

        if (self._curr_steps >= self._ref_motion_frames) or dones[0]:
            # trim the frames
            self.frames["robot_state"]["root_pos"] = self.frames["robot_state"]["root_pos"][:self._curr_steps, :]
            self.frames["robot_state"]["root_lin_vel"] = self.frames["robot_state"]["root_lin_vel"][:self._curr_steps, :]
            self.frames["robot_state"]["root_ang_vel"] = self.frames["robot_state"]["root_ang_vel"][:self._curr_steps, :]
            self.frames["robot_state"]["root_rot"] = self.frames["robot_state"]["root_rot"][:self._curr_steps, :]
            self.frames["robot_state"]["joint_pos"] = self.frames["robot_state"]["joint_pos"][:self._curr_steps, :]
            self.frames["robot_state"]["joint_vel"] = self.frames["robot_state"]["joint_vel"][:self._curr_steps, :]
            self.frames["action"] = self.frames["action"][:self._curr_steps, :]
            self.frames["step"] = self._curr_steps

            self._recorded_frames_total.append(self.frames)
            self._curr_steps = 0
            self._num_episodes += 1
            self._pbar.update(1)
            if self._num_episodes >= self._num_unique_ref_motions // self._num_envs:
                return True
            else:
                self.frames = None
        return False
    
    def save(self, name="test"):
        """Save the recorded data to the specified log path."""
        log_path = f"{self.log_path}/{name}.h5"
        with h5py.File(log_path, "w") as f:
            f.attrs["size"] = len(self._recorded_frames_total)

            for idx, item in enumerate(self._recorded_frames_total):
                group = f.create_group(f"episode_{idx}")
                group.attrs["len"] = item["step"]
                group.create_dataset("reference_motion_id", data=item["ref_motion_id"])
                group.create_dataset("mask_mode", data=item["mask"], compression="gzip")
                group.create_dataset("action", data=item["action"], compression="gzip")
                
                robot_state_group = group.create_group("robot_state")
                robot_state_group.create_dataset("root_pos", data=item["robot_state"]["root_pos"], compression="gzip")
                robot_state_group.create_dataset("root_lin_vel", data=item["robot_state"]["root_lin_vel"], compression="gzip")
                robot_state_group.create_dataset("root_ang_vel", data=item["robot_state"]["root_ang_vel"], compression="gzip")
                robot_state_group.create_dataset("root_rot", data=item["robot_state"]["root_rot"], compression="gzip")
                robot_state_group.create_dataset("joint_pos", data=item["robot_state"]["joint_pos"], compression="gzip")
                robot_state_group.create_dataset("joint_vel", data=item["robot_state"]["joint_vel"], compression="gzip")
        print(f"Recorded data saved to {log_path}")
        self._pbar.close()



        