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

import numpy as np
import pytest
import torch

from lerobot.processor import Numpy2TorchActionProcessorStep, Torch2NumpyActionProcessorStep, TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY


@pytest.mark.parametrize(
    "shape, squeeze, expected",
    [((1, 7), True, (7,)), ((1, 7), False, (1, 7)), ((3, 7), True, (3, 7)), ((1, 1, 7), True, (1, 7))],
)
def test_torch2numpy_only_squeezes_a_leading_singleton(shape, squeeze, expected):
    step = Torch2NumpyActionProcessorStep(squeeze_batch_dim=squeeze)

    out = step(create_transition(action=torch.zeros(*shape, requires_grad=True)))[TransitionKey.ACTION]

    assert isinstance(out, np.ndarray)
    assert out.shape == expected


def test_torch2numpy_rejects_non_tensor_action():
    with pytest.raises(TypeError, match="Expected PolicyAction"):
        Torch2NumpyActionProcessorStep()(create_transition(action=np.zeros(3)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_numpy2torch_preserves_dtype(dtype):
    action = np.arange(4).astype(dtype)

    out = Numpy2TorchActionProcessorStep()(create_transition(action=action))[TransitionKey.ACTION]

    assert out.dtype == torch.from_numpy(action).dtype
    np.testing.assert_array_equal(out.numpy(), action)


def test_numpy2torch_passes_through_a_missing_action():
    assert Numpy2TorchActionProcessorStep()(create_transition(action=None))[TransitionKey.ACTION] is None


def test_numpy2torch_rejects_non_numpy_action():
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        Numpy2TorchActionProcessorStep()(create_transition(action=torch.zeros(3)))


def test_numpy2torch_converts_only_numpy_teleop_actions():
    step = Numpy2TorchActionProcessorStep()
    action = np.zeros(2, dtype=np.float32)

    converted = step(
        create_transition(action=action, complementary_data={TELEOP_ACTION_KEY: np.array([1.0, 2.0])})
    )[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]
    teleop_dict = {"delta_x": 0.1}
    untouched = step(create_transition(action=action, complementary_data={TELEOP_ACTION_KEY: teleop_dict}))[
        TransitionKey.COMPLEMENTARY_DATA
    ][TELEOP_ACTION_KEY]

    assert converted.dtype == torch.float32
    assert converted.tolist() == [1.0, 2.0]
    assert untouched is teleop_dict
