import numpy as np

from kiox.offline import (
    create_frame_stack_kiox_from_dataset,
    create_simple_kiox_from_dataset,
)


def test_create_simple_kiox_from_dataset():
    observations = np.random.random((1000, 100))
    actions = np.random.random((1000, 4))
    rewards = np.random.random(1000)
    terminals = np.zeros(1000)

    kiox = create_simple_kiox_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    assert kiox.step_buffer.size() == 1000
    assert kiox.transition_buffer.size() == 999


def test_create_frame_stack_kiox_from_dataset():
    observations = np.random.random((1000, 1, 84, 84))
    actions = np.random.random((1000, 4))
    rewards = np.random.random(1000)
    terminals = np.zeros(1000)

    kiox = create_frame_stack_kiox_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        n_frames=3,
    )

    assert kiox.step_buffer.size() == 1000
    assert kiox.transition_buffer.size() == 999

    batch = kiox.sample(32)
    assert batch.observations.shape == (32, 3, 84, 84)
    assert batch.next_observations.shape == (32, 3, 84, 84)
