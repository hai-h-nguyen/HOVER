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


import pprint
import time
import torch
import yaml

from hw_wrappers.unitree_h1 import UnitreeH1  # noqa
from inference_env.deployment_player import DeploymentPlayer
from inference_env.neural_wbc_env_cfg_real_h1 import NeuralWBCEnvCfgRealH1
from inference_env.utils import get_player_args

from neural_wbc.data import get_data_path

parser = get_player_args(description="Basic deployment player for running HOVER policy on real robots.")
args_cli = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    custom_config = None
    if args_cli.env_config_overwrite is not None:
        with open(args_cli.env_config_overwrite) as fh:
            custom_config = yaml.safe_load(fh)
        print("[INFO]: Using custom configuration:")
        pprint.pprint(custom_config)

    env_cfg = NeuralWBCEnvCfgRealH1(model_xml_path=get_data_path("mujoco/models/scene.xml"))

    player = DeploymentPlayer(
        args_cli=args_cli,
        env_cfg=env_cfg,
        custom_config=custom_config,
    )

    inference_time = env_cfg.decimation * env_cfg.dt
    print("Deploying policy on real robot.")
    start_time = time.time()
    elapsed_time = 0.0
    for i in range(args_cli.max_iterations):
        print(
            "\rActual loop frequency: {:.2f} Hz | Update time: {:.2f}s".format(
                1 / (time.time() - start_time), elapsed_time
            ),
            end="",
            flush=True,
        )
        start_time = time.time()
        player.play_once()
        elapsed_time = time.time() - start_time
        remaining_time = inference_time - elapsed_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)


if __name__ == "__main__":
    main()
