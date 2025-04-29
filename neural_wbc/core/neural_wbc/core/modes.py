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
import enum


class NeuralWBCModes(enum.Enum):
    TRAIN = 0
    TEST = 1
    DISTILL = 2
    DISTILL_TEST = 3
    DELTA_ACTION = 4
    DELTA_ACTION_TEST = 5
    FINETUNE = 6
    FINETUNE_TEST = 7

    def is_distill_mode(self):
        return self in {self.DISTILL, self.DISTILL_TEST}

    def is_distill_test_mode(self):
        return self == self.DISTILL_TEST
    
    def is_delta_action_mode(self):
        return self in {self.DELTA_ACTION, self.DELTA_ACTION_TEST}
    
    def is_delta_action_test_mode(self):
        return self == self.DELTA_ACTION_TEST
    
    def is_finetune_mode(self):
        return self in {self.FINETUNE, self.FINETUNE_TEST}

    def is_finetune_test_mode(self):
        return self == self.FINETUNE_TEST

    def is_training_mode(self):
        return self in {self.TRAIN, self.DISTILL, self.DELTA_ACTION, self.FINETUNE}

    def is_test_mode(self):
        return self in {self.TEST, self.DISTILL_TEST, self.DELTA_ACTION_TEST, self.FINETUNE_TEST}