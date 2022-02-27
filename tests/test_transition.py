from collections import deque

import numpy as np

from kiox.episode import EpisodeManager
from kiox.transition import FrameStackLazyTransition, SimpleLazyTransition

from .utility import StepFactory


def test_simple_lazy_transition():
    factory = StepFactory()
    episode_manager = EpisodeManager()

    step1 = factory()
    episode_manager.append(step1)

    step2 = factory(terminal=True)
    episode_manager.append(step2)

    # test transition
    lazy_transition1 = SimpleLazyTransition(
        curr_idx=step1.idx,
        next_idx=step2.idx,
        multi_step_reward=1.0,
        duration=1,
    )
    transition = lazy_transition1.create(episode_manager)
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
    transition = lazy_transition2.create(episode_manager)
    assert np.all(transition.observation == step2.observation)
    assert np.all(transition.next_observation == 0.0)
    assert transition.reward == 1.0
    assert transition.terminal == 1.0
    assert transition.duration == 1


def test_frame_stack_lazy_transition():
    factory = StepFactory(observation_shape=(1, 84, 84))
    episode_manager = EpisodeManager()
    steps = []
    prev_idx = deque(maxlen=4)
    frames = deque(maxlen=4)

    for _ in range(4):
        frames.append(np.zeros((1, 84, 84)))

    for i in range(9):
        step = factory()
        episode_manager.append(step)
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
            transition = lazy_transition.create(episode_manager)

            ref_observation = np.vstack(list(frames)[:-1])
            ref_next_observation = np.vstack(list(frames)[1:])

            assert transition.observation.shape == (3, 84, 84)
            assert transition.next_observation.shape == (3, 84, 84)
            assert np.all(transition.observation == ref_observation)
            assert np.all(transition.next_observation == ref_next_observation)
            assert transition.reward == 1.0
            assert transition.terminal == 0.0
            assert transition.duration == 1

    step = factory(terminal=True)
    episode_manager.append(step)
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
    transition = lazy_transition.create(episode_manager)

    ref_observation = np.vstack(list(frames)[1:])
    ref_next_observation = np.zeros((3, 84, 84))

    assert np.all(transition.observation == ref_observation)
    assert np.all(transition.next_observation == ref_next_observation)
    assert transition.reward == 1.0
    assert transition.terminal == 1.0
    assert transition.duration == 1
