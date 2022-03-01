import pytest

from kiox.batch_factory import BatchFactory
from kiox.transition_buffer import UnlimitedTransitionBuffer

from .utility import StepFactory, TransitionFactory


@pytest.mark.parametrize("observation_shape", [(100,), (3, 84, 84)])
def test_batch_factory(observation_shape):
    factory = TransitionFactory(StepFactory(observation_shape))
    buffer = UnlimitedTransitionBuffer()

    for _ in range(100):
        transition = factory()
        buffer.append(transition)

    batch_factory = BatchFactory(factory.step_buffer, buffer)

    batch = batch_factory.sample(32)
    assert batch.observations.shape == (32,) + observation_shape
    assert batch.actions.shape == (32, 4)
    assert batch.rewards.shape == (32, 1)
    assert batch.terminals.shape == (32, 1)
    assert batch.durations.shape == (32, 1)
