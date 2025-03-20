from phc.utils.motion_lib_base import MotionLibBase
from phc.utils.torch_g1_humanoid_batch import Humanoid_Batch
from enum import Enum

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionLibG1(MotionLibBase):
    def __init__(self, 
                 motion_file, 
                 device,
                 num_envs, 
                 skeleton_file="resources/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml",
                 fix_height=FixHeightMode.no_fix, 
                 multi_thread=True, 
                 extend_hand = True, 
                 extend_head = False, 
                 sim_timestep = 1/50):
        super().__init__(motion_file=motion_file, device=device, num_envs=num_envs, skeleton_file=skeleton_file, fix_height=fix_height, multi_thread=multi_thread, sim_timestep = sim_timestep)
        self.mesh_parsers = Humanoid_Batch(extend_hand = extend_hand, extend_head = extend_head, mjcf_file=skeleton_file)
        return
