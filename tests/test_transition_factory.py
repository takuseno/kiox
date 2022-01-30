from kiox.step_buffer import EpisodicStepBuffer
from kiox.transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
)

from .utility import StepFactory


def test_simple_transition_factory():
    factory = StepFactory()
    buffer = EpisodicStepBuffer()
    steps = []

    for _ in range(10):
        step = factory()
        buffer.append(step)
        steps.append(step)

    transition_factory = SimpleTransitionFactory()

    for i in range(10):
        if i == 9:
            lazy_transition = transition_factory.create(
                step=steps[i],
                next_step=None,
                episode_steps=buffer,
                duration=1,
                gamma=0.99,
            )
            assert lazy_transition.next_idx is None
        else:
            lazy_transition = transition_factory.create(
                step=steps[i],
                next_step=steps[i + 1],
                episode_steps=buffer,
                duration=1,
                gamma=0.99,
            )
            assert lazy_transition.next_idx is steps[i + 1].idx

        assert lazy_transition.curr_idx is steps[i].idx
        assert lazy_transition.multi_step_reward == steps[i].reward
        assert lazy_transition.duration == 1


def test_frame_stack_transition_factory():
    factory = StepFactory(observation_shape=(1, 84, 84))
    buffer = EpisodicStepBuffer()
    steps = []

    for _ in range(10):
        step = factory()
        buffer.append(step)
        steps.append(step)

    transition_factory = FrameStackTransitionFactory(n_frames=3)

    for i in range(10):
        if i == 9:
            lazy_transition = transition_factory.create(
                step=steps[i],
                next_step=None,
                episode_steps=buffer,
                duration=1,
                gamma=0.99,
            )
            assert lazy_transition.next_idx is None
        else:
            lazy_transition = transition_factory.create(
                step=steps[i],
                next_step=steps[i + 1],
                episode_steps=buffer,
                duration=1,
                gamma=0.99,
            )
            assert lazy_transition.next_idx is steps[i + 1].idx

        prev_frames = [step.idx for step in steps[max(i - 2, 0) : i]]

        assert lazy_transition.curr_idx is steps[i].idx
        assert lazy_transition.multi_step_reward == steps[i].reward
        assert lazy_transition.duration == 1
        assert lazy_transition.prev_frames == prev_frames
