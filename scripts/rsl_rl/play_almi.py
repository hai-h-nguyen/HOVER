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


import torch

from almi_policy_cfg import ALMIPolicyCfg

from isaaclab.app import AppLauncher
from isaaclab.devices import Se2Keyboard

# local imports
from utils import get_player_args  # isort: skip


# add argparse arguments
parser = get_player_args(description="Plays motion tracking policy in Isaac Lab.")
parser.add_argument("--randomize", action="store_true", help="Whether to randomize reference motion while playing.")

# append RSL-RL cli arguments
ALMIPolicyCfg.add_args_to_parser(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from almi_players import DemoPlayer


def main():
    # run the main execution
    keyboard_interface = Se2Keyboard(v_x_sensitivity=0.1, v_y_sensitivity=0.1, omega_z_sensitivity=0.1)

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    keyboard_interface.add_callback("R", reset_recording_instance)
    print(keyboard_interface)

    player = DemoPlayer(args_cli=args_cli, randomize=args_cli.randomize)

    player.wrapped_env.reset()
    keyboard_interface.reset()
    obs = player.wrapped_env.get_observations()
    player.env.commands = torch.zeros_like(player.env.commands)

    while simulation_app.is_running() and not player._should_stop():
        print(player.env.commands)
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard commands
            delta_commands = keyboard_interface.advance()
            player.env.commands[:, 0] += delta_commands[0]
            player.env.commands[:, 1] += delta_commands[1]
            player.env.commands[:, 3] += delta_commands[2]
            # agent stepping
            actions = player.policy(obs)
            # env stepping
            obs, privileged_obs, rewards, dones, extras = player.wrapped_env.step(actions)
            obs = player._post_step(obs=obs, privileged_obs=privileged_obs, rewards=rewards, dones=dones, extras=extras)

            if should_reset_recording_instance:
                player.wrapped_env.reset()
                player.env.commands = torch.zeros_like(player.env.commands)
                should_reset_recording_instance = False

    # close the simulator
    player.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
