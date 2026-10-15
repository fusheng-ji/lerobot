#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from lerobot.processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep, TransitionKey
from lerobot.processor.converters import create_transition


def _delta(dx=0.0, dy=0.0, dz=0.0, gripper=1.0):
    return {"delta_x": dx, "delta_y": dy, "delta_z": dz, "gripper": gripper}


@pytest.mark.parametrize("use_gripper", [True, False])
def test_tensor_to_delta_dict(use_gripper):
    step = MapTensorToDeltaActionDictStep(use_gripper=use_gripper)

    # A leading batch dim is squeezed; the 4th element is only read when use_gripper is set.
    action = step(create_transition(action=torch.tensor([[0.1, -0.2, 0.3, 2.0]])))[TransitionKey.ACTION]

    expected = {"delta_x": 0.1, "delta_y": -0.2, "delta_z": 0.3} | ({"gripper": 2.0} if use_gripper else {})
    assert action == pytest.approx(expected)


def test_tensor_to_delta_dict_rejects_non_tensor_action():
    with pytest.raises(ValueError, match="Only PolicyAction"):
        MapTensorToDeltaActionDictStep()(create_transition(action=_delta()))


def test_delta_to_robot_action_scales_position_and_zeroes_rotation():
    step = MapDeltaActionToRobotActionStep(position_scale=0.5)
    original = _delta(dx=0.2, dy=-0.4, dz=0.6, gripper=2)

    action = step(create_transition(action=original))[TransitionKey.ACTION]

    assert action == pytest.approx(
        {
            "enabled": True,
            "target_x": 0.1,
            "target_y": -0.2,
            "target_z": 0.3,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": 2.0,
        }
    )
    assert original == _delta(dx=0.2, dy=-0.4, dz=0.6, gripper=2)  # the input dict is left alone


def test_delta_to_robot_action_enabled_follows_the_unscaled_norm():
    # The scale is applied after the noise check, so it must not change whether the step is enabled.
    step = MapDeltaActionToRobotActionStep(position_scale=0.001, noise_threshold=0.05)

    below = step(create_transition(action=_delta(dx=0.03, dy=0.03)))[TransitionKey.ACTION]  # norm ~0.042
    above = step(create_transition(action=_delta(dx=0.03, dy=0.05)))[TransitionKey.ACTION]  # norm ~0.058

    assert not below["enabled"]
    assert above["enabled"]


def test_delta_to_robot_action_requires_dict_action():
    with pytest.raises(ValueError, match="RobotAction"):
        MapDeltaActionToRobotActionStep()(create_transition(action=torch.zeros(4)))
