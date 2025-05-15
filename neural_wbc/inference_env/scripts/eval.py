# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import pprint
import yaml

from inference_env.deployment_player import DeploymentPlayer
from inference_env.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1
from inference_env.neural_wbc_env_cfg_h12 import NeuralWBCEnvCfgH12
from inference_env.neural_wbc_env_cfg_g1 import NeuralWBCEnvCfgG1
from inference_env.utils import get_player_args

from neural_wbc.core.evaluator import Evaluator
from neural_wbc.core.recorder import Recorder
from neural_wbc.data import get_data_path

# add argparse arguments
parser = get_player_args(description="Evaluates motion tracking policy and collects metrics in MuJoCo.")
parser.add_argument("--metrics_path", type=str, default=None, help="Path to store metrics in.")
parser.add_argument("--robot_model", type=str, choices=["h1", "h12", "g1", "gr1"], default="h1", help="Robot used in environment")
parser.add_argument("--record", action="store_true", help="Record the simulation.")
args_cli = parser.parse_args()


def main():
    custom_config = None
    if args_cli.env_config_overwrite is not None:
        with open(args_cli.env_config_overwrite) as fh:
            custom_config = yaml.safe_load(fh)
        print("[INFO]: Using custom configuration:")
        pprint.pprint(custom_config)

    # Initialize the player with the updated properties
    if args_cli.robot_model == "h1":
        env_cfg=NeuralWBCEnvCfgH1(model_xml_path=get_data_path("mujoco/models/h1/scene.xml"))
    elif args_cli.robot_model == "h12":
        env_cfg=NeuralWBCEnvCfgH12(model_xml_path=get_data_path("mujoco/models/h12/scene.xml"))
    elif args_cli.robot_model == "g1":
        env_cfg=NeuralWBCEnvCfgG1(model_xml_path=get_data_path("mujoco/models/g1/scene.xml"))
    else:
        raise ValueError
    
    player = DeploymentPlayer(
        args_cli=args_cli,
        custom_config=custom_config,
        env_cfg=env_cfg,
        )
    evaluator = Evaluator(env_wrapper=player.env, metrics_path=args_cli.metrics_path)
    if args_cli.record:
        recorder = Recorder(env_wrapper=player.env, log_path="/home/rtx4/HOVER/")
    else:
        recorder = None

    should_stop = False
    is_record = False
    while not should_stop:
        _, obs, dones, extras = player.play_once()

        if recorder is not None:
            is_record = recorder.collect(dones=dones.clone(), info=extras.copy())
        reset_env = evaluator.collect(dones=dones, info=extras)
        if reset_env and not evaluator.is_evaluation_complete():
            # evaluator.visualize(dt=env_cfg.dt * env_cfg.decimation)
            evaluator.forward_motion_samples()
            _ = player.reset()
        should_stop = evaluator.is_evaluation_complete()

    evaluator.conclude()
    if recorder is not None:
        recorder.save()
        print("Recording completed and saved.")
    else:
        print("No recording was made.")


if __name__ == "__main__":
    main()
