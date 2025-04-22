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
import torch.nn as nn
from torch.distributions import Normal

class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.weight_pos_embed = nn.Embedding(context_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)

        # take the last token
        x = x[:, -1, :]
        x = self.output_layer(x)

        return x

class StudentTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, distilled_imitation_dim=168, latent_dim=128,num_head=4, num_layer=4, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.distilled_imitation_dim = distilled_imitation_dim
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.dropout_rate = dropout_rate

        self.single_len = int((input_dim - distilled_imitation_dim) / (context_len+1))
        # Transformer model for distilled imitation learning
        self.transformer = Transformer(self.single_len, output_dim, context_len+2, latent_dim, num_head, num_layer, dropout_rate)
        self.input_layer = nn.Sequential(nn.Linear(distilled_imitation_dim, self.single_len),
                                         nn.Dropout(dropout_rate))
        
    def forward(self, x):
        # Indices for slicing
        robot_state_end = self.single_len - self.output_dim
        imitation_end = robot_state_end + self.distilled_imitation_dim
        last_action_end = self.single_len + self.distilled_imitation_dim

        # Slice inputs
        x_distilled_robot_state = x[:, :robot_state_end]
        x_distilled_imitation = x[:, robot_state_end:imitation_end]
        x_last_action = x[:, imitation_end:last_action_end]
        x_history = x[:, last_action_end:]

        # Reshape history and transpose to (batch, context_len, single_len)
        x_history = x_history.view(-1, self.single_len, self.context_len).transpose(1, 2)

        # Prepare current state and imitation embeddings
        x_current = torch.cat([x_distilled_robot_state, x_last_action], dim=1).unsqueeze(1)
        x_imitation = self.input_layer(x_distilled_imitation).unsqueeze(1)

        # Concatenate along sequence dimension: [current, imitation, history]
        x = torch.cat([x_current, x_imitation, x_history], dim=1)

        # Pass through transformer
        return self.transformer(x)

class StudentPolicy(nn.Module):
    def __init__(self, 
                 num_obs, 
                 num_actions, 
                 policy_hidden_dims=[256, 256, 256], 
                 activation="elu", 
                 noise_std=0.001,
                ):
        super().__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_obs

        # Policy
        policy_layers = []
        policy_layers.append(nn.Linear(mlp_input_dim_a, policy_hidden_dims[0]))
        policy_layers.append(activation)
        for layer in range(len(policy_hidden_dims)):
            if layer == len(policy_hidden_dims) - 1:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer], num_actions))
            else:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer], policy_hidden_dims[layer + 1]))
                policy_layers.append(activation)
        self.policy = nn.Sequential(*policy_layers)

        print(f"Student Policy MLP: {self.policy}")

        # Action noise
        self.std = nn.Parameter(noise_std * torch.ones(num_actions))
        self.std.requires_grad = False
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.policy(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.policy(observations)
        return actions_mean

    def load(self, path, device):
        self.to(device)
        loaded_dict = torch.load(path, map_location=device)
        self.load_state_dict(loaded_dict["model_state_dict"])


def get_activation(act_name):
    activation_functions = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    return activation_functions.get(act_name, lambda: print(f"invalid activation function {act_name}!"))()
