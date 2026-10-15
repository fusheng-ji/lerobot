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

from lerobot.processor import (
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.hil_processor import (
    DISCRETE_PENALTY_KEY,
    TELEOP_ACTION_KEY,
    GripperPenaltyProcessorStep,
)
from lerobot.teleoperators.utils import TeleopEvents

# --- ImageCropResizeProcessorStep ---


def test_image_crop_resize_crops_only_configured_keys_then_resizes_and_clamps():
    front, wrist = torch.full((3, 16, 16), 5.0), torch.rand(3, 16, 16)
    observation = {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": 0,
    }

    cropped = ImageCropResizeProcessorStep(
        crop_params_dict={"observation.images.front": (2, 3, 4, 5)}
    ).observation(observation)
    resized = ImageCropResizeProcessorStep(resize_size=(4, 6)).observation(observation)

    assert cropped["observation.images.front"].shape == (3, 4, 5)
    assert cropped["observation.images.wrist"] is wrist
    assert cropped["observation.state"] == 0
    assert resized["observation.images.wrist"].shape == (3, 4, 6)
    assert resized["observation.images.front"].max() <= 1.0  # out-of-range input is clamped to [0, 1]
    assert observation["observation.images.front"] is front  # the input is not modified


# --- TimeLimitProcessorStep ---


def test_time_limit_truncates_at_max_steps_and_reset_restarts_the_count():
    step = TimeLimitProcessorStep(max_episode_steps=3)

    flags = [step(create_transition(truncated=False))[TransitionKey.TRUNCATED] for _ in range(4)]
    step.reset()

    assert flags == [False, False, True, True]
    assert step(create_transition(truncated=False))[TransitionKey.TRUNCATED] is False
    assert step(create_transition(truncated=True))[TransitionKey.TRUNCATED] is True  # incoming flag is kept


# --- GymHILAdapterProcessorStep ---


def test_gym_hil_adapter_copies_info_fields_into_complementary_data_and_events():
    info = {TELEOP_ACTION_KEY: np.array([1.0]), DISCRETE_PENALTY_KEY: -0.02, "is_intervention": True}

    out = GymHILAdapterProcessorStep()(create_transition(info=info, complementary_data={"other": 1}))

    assert out[TransitionKey.COMPLEMENTARY_DATA] == {
        "other": 1,
        TELEOP_ACTION_KEY: info[TELEOP_ACTION_KEY],
        DISCRETE_PENALTY_KEY: -0.02,
    }
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True


# --- GripperPenaltyProcessorStep ---


def _gripper_transition(gripper_pos, gripper_action):
    return create_transition(
        action=torch.tensor([0.0, 0.0, 0.0, gripper_action]),
        complementary_data={"raw_joint_positions": {"gripper.pos": gripper_pos}},
    )


@pytest.mark.parametrize(
    "gripper_pos, gripper_action, expected",
    [
        (0.0, 30.0, -0.02),  # open, commanded closed
        (30.0, 0.0, -0.02),  # closed, commanded open
        (0.0, 0.0, 0.0),  # open, stays open
        (30.0, 30.0, 0.0),  # closed, stays closed
        (15.0, 30.0, 0.0),  # half way is neither open nor closed
    ],
)
def test_gripper_penalty_only_fires_on_extreme_to_extreme_transitions(gripper_pos, gripper_action, expected):
    out = GripperPenaltyProcessorStep()(_gripper_transition(gripper_pos, gripper_action))

    assert out[TransitionKey.COMPLEMENTARY_DATA][DISCRETE_PENALTY_KEY] == pytest.approx(expected)


def test_gripper_penalty_is_a_noop_without_a_gripper_reading_or_action():
    step = GripperPenaltyProcessorStep()
    no_gripper = create_transition(action=torch.zeros(4), complementary_data={"raw_joint_positions": {}})
    no_action = create_transition(
        action=None, complementary_data={"raw_joint_positions": {"gripper.pos": 1.0}}
    )

    for transition in (create_transition(action=torch.zeros(4)), no_gripper, no_action):
        assert DISCRETE_PENALTY_KEY not in step(transition)[TransitionKey.COMPLEMENTARY_DATA]


def test_gripper_penalty_rejects_dict_actions():
    transition = create_transition(
        action={"gripper": 1.0}, complementary_data={"raw_joint_positions": {"gripper.pos": 0.0}}
    )

    with pytest.raises(ValueError, match="tensor or array action"):
        GripperPenaltyProcessorStep()(transition)


# --- InterventionActionProcessorStep ---


def _intervention_transition(*, action=None, teleop_action=None, **events):
    return create_transition(
        action=torch.zeros(3) if action is None else action,
        info={TeleopEvents(name): value for name, value in events.items()},
        complementary_data={} if teleop_action is None else {TELEOP_ACTION_KEY: teleop_action},
    )


def test_intervention_keeps_the_policy_action_when_not_intervening():
    policy_action = torch.tensor([0.1, 0.2, 0.3])

    out = InterventionActionProcessorStep()(
        _intervention_transition(action=policy_action, teleop_action={"delta_x": 1.0})
    )

    assert out[TransitionKey.ACTION] is policy_action
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is False


def test_intervention_requires_tensor_action():
    with pytest.raises(ValueError, match="PolicyAction"):
        InterventionActionProcessorStep()(create_transition(action={"delta_x": 0.0}))


@pytest.mark.parametrize("use_gripper, expected", [(False, [0.1, 0.0, 0.3]), (True, [0.1, 0.0, 0.3, 1.0])])
def test_intervention_overrides_action_with_a_teleop_dict(use_gripper, expected):
    step = InterventionActionProcessorStep(use_gripper=use_gripper)

    out = step(_intervention_transition(is_intervention=True, teleop_action={"delta_x": 0.1, "delta_z": 0.3}))

    assert out[TransitionKey.ACTION].tolist() == pytest.approx(expected)  # missing gripper defaults to 1.0
    assert out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY] is out[TransitionKey.ACTION]


@pytest.mark.filterwarnings("ignore:To copy construct from a tensor")
@pytest.mark.parametrize(
    "teleop_action", [np.array([0.1, 0.2, 0.3]), torch.tensor([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]]
)
def test_intervention_accepts_array_like_teleop_actions(teleop_action):
    policy_action = torch.zeros(3, dtype=torch.float32)

    out = InterventionActionProcessorStep()(
        _intervention_transition(action=policy_action, is_intervention=True, teleop_action=teleop_action)
    )

    assert out[TransitionKey.ACTION].dtype == policy_action.dtype
    assert out[TransitionKey.ACTION].tolist() == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "events, terminate_on_success, expected_done, expected_reward",
    [
        ({"success": True}, True, True, 1.0),
        ({"success": True}, False, False, 1.0),
        ({"terminate_episode": True}, True, True, 0.0),
        ({}, True, False, 0.0),
    ],
)
def test_intervention_done_and_reward(events, terminate_on_success, expected_done, expected_reward):
    step = InterventionActionProcessorStep(terminate_on_success=terminate_on_success)

    out = step(_intervention_transition(**events))

    assert out[TransitionKey.DONE] is expected_done
    assert out[TransitionKey.REWARD] == expected_reward


# --- RewardClassifierProcessorStep ---


class FakeClassifier:
    def __init__(self, success):
        self.success = success
        self.seen_keys = None

    def predict_reward(self, images, threshold=0.5):
        self.seen_keys = list(images)
        return torch.tensor([float(self.success)])


def _classifier_transition(**kwargs):
    observation = {"observation.images.front": torch.rand(1, 3, 4, 4), "observation.state": torch.zeros(1, 6)}
    return create_transition(observation=observation, **kwargs)


@pytest.mark.parametrize(
    "success, terminate_on_success, expected_reward, expected_done",
    [(True, True, 5.0, True), (True, False, 5.0, False), (False, True, 0.25, False)],
)
def test_reward_classifier_updates_reward_and_done(
    success, terminate_on_success, expected_reward, expected_done
):
    classifier = FakeClassifier(success)
    step = RewardClassifierProcessorStep(
        reward_classifier=classifier, success_reward=5.0, terminate_on_success=terminate_on_success
    )

    out = step(_classifier_transition(reward=0.25, done=False))

    assert out[TransitionKey.REWARD] == expected_reward
    assert out[TransitionKey.DONE] is expected_done
    assert classifier.seen_keys == ["observation.images.front"]  # only image keys reach the classifier


def test_reward_classifier_is_a_noop_without_a_model_or_images():
    classifier = FakeClassifier(success=True)
    no_images = create_transition(observation={"observation.state": torch.zeros(1, 6)})

    assert RewardClassifierProcessorStep()(_classifier_transition())[TransitionKey.REWARD] == 0.0
    assert RewardClassifierProcessorStep(reward_classifier=classifier)(no_images)[TransitionKey.REWARD] == 0.0
    assert classifier.seen_keys is None
