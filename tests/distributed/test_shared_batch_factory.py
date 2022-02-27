import numpy as np
import pytest

from kiox.distributed.shared_batch_factory import SharedBatchFactory
from kiox.transition_buffer import UnlimitedTransitionBuffer

from ..utility import StepFactory, TransitionFactory


@pytest.mark.parametrize("observation_shape", [(100,), (3, 84, 84)])
def test_shared_batch_factory(observation_shape):
    factory = TransitionFactory(StepFactory(observation_shape))
    buffer = UnlimitedTransitionBuffer()

    for _ in range(100):
        transition = factory()
        buffer.append(transition)

    batch_factory = SharedBatchFactory(observation_shape, (4,), (1,), 32)

    init_observations = batch_factory.batch.observations.copy()
    init_next_observations = batch_factory.batch.next_observations.copy()
    init_actions = batch_factory.batch.actions.copy()
    init_rewards = batch_factory.batch.rewards.copy()
    init_durations = batch_factory.batch.durations.copy()

    batch_factory.sample(factory.episode_manager, buffer)
    observations = batch_factory.batch.observations.copy()
    next_observations = batch_factory.batch.next_observations.copy()
    actions = batch_factory.batch.actions.copy()
    rewards = batch_factory.batch.rewards.copy()
    terminals = batch_factory.batch.terminals.copy()
    durations = batch_factory.batch.durations.copy()

    # check shape
    assert observations.shape == (32, *observation_shape)
    assert next_observations.shape == (32, *observation_shape)
    assert actions.shape == (32, 4)
    assert rewards.shape == (32, 1)
    assert terminals.shape == (32, 1)
    assert durations.shape == (32, 1)

    # check if everything is changed
    assert np.all(observations != init_observations)
    assert np.all(next_observations != init_next_observations)
    assert np.all(actions != init_actions)
    assert np.all(rewards != init_rewards)
    assert np.all(durations != init_durations)
