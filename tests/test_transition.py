from collections import deque

import numpy as np

from kiox.episode import EpisodeManager
from kiox.step import StepBuffer
from kiox.transition import FrameStackLazyTransition, SimpleLazyTransition
from kiox.transition_buffer import UnlimitedTransitionBuffer

from .utility import StepFactory


def test_simple_lazy_transition():
    factory = StepFactory()
    step_buffer = StepBuffer()
    episode_manager = EpisodeManager(step_buffer, UnlimitedTransitionBuffer())

    step1 = episode_manager.append_step(factory())

    step2 = episode_manager.append_step(factory(terminal=True))

    # test transition
    lazy_transition1 = SimpleLazyTransition(
        curr_idx=step1.idx,
        next_idx=step2.idx,
        multi_step_reward=1.0,
        duration=1,
    )
    transition = lazy_transition1.create(step_buffer)
    assert np.all(transition.observation == step1.observation)
    assert np.all(transition.next_observation == step2.observation)
    assert transition.reward == 1.0
    assert transition.terminal == 0.0
    assert transition.duration == 1

    # test terminal transition
    lazy_transition2 = SimpleLazyTransition(
        curr_idx=step2.idx,
        next_idx=None,
        multi_step_reward=1.0,
        duration=1,
    )
    transition = lazy_transition2.create(step_buffer)
    assert np.all(transition.observation == step2.observation)
    assert np.all(transition.next_observation == 0.0)
    assert transition.reward == 1.0
    assert transition.terminal == 1.0
    assert transition.duration == 1


def test_frame_stack_lazy_transition():
    factory = StepFactory(observation_shape=(1, 84, 84))
    step_buffer = StepBuffer()
    episode_manager = EpisodeManager(step_buffer, UnlimitedTransitionBuffer())
    steps = []
    prev_idx = deque(maxlen=4)
    frames = deque(maxlen=4)

    for _ in range(4):
        frames.append(np.zeros((1, 84, 84)))

    for i in range(9):
        step = episode_manager.append_step(factory())
        steps.append(step)
        prev_idx.append(step.idx)
        frames.append(step.observation)

        if i > 0:
            lazy_transition = FrameStackLazyTransition(
                curr_idx=steps[i - 1].idx,
                next_idx=steps[i].idx,
                multi_step_reward=1.0,
                duration=1,
                n_frames=3,
                prev_frames=list(prev_idx)[:-2],
            )
            transition = lazy_transition.create(step_buffer)

            ref_observation = np.vstack(list(frames)[:-1])
            ref_next_observation = np.vstack(list(frames)[1:])

            assert transition.observation.shape == (3, 84, 84)
            assert transition.next_observation.shape == (3, 84, 84)
            assert np.all(transition.observation == ref_observation)
            assert np.all(transition.next_observation == ref_next_observation)
            assert transition.reward == 1.0
            assert transition.terminal == 0.0
            assert transition.duration == 1

    step = episode_manager.append_step(factory(terminal=True))
    steps.append(step)
    prev_idx.append(step.idx)
    frames.append(step.observation)

    # test terminal transition
    lazy_transition = FrameStackLazyTransition(
        curr_idx=steps[-1].idx,
        next_idx=None,
        multi_step_reward=1.0,
        duration=1,
        n_frames=3,
        prev_frames=list(prev_idx)[1:-1],
    )
    transition = lazy_transition.create(step_buffer)

    ref_observation = np.vstack(list(frames)[1:])
    ref_next_observation = np.zeros((3, 84, 84))

    assert np.all(transition.observation == ref_observation)
    assert np.all(transition.next_observation == ref_next_observation)
    assert transition.reward == 1.0
    assert transition.terminal == 1.0
    assert transition.duration == 1
