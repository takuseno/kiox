import numpy as np

from kiox.offline import (
    create_frame_stack_kiox_from_dataset,
    create_simple_kiox_from_dataset,
)


def test_create_simple_kiox_from_dataset_ndarray():
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

    assert kiox.episode_manager.get_total_step_size() == 1000
    assert kiox.transition_buffer.size() == 999

    batch = kiox.sample(32)
    assert batch.observations.shape == (32, 100)
    assert batch.actions.shape == (32, 4)
    assert batch.next_observations.shape == (32, 100)


def test_create_simple_kiox_from_dataset_tuple():
    observations = (
        np.random.random((1000, 100)),
        np.random.random((1000, 3, 84, 84)),
    )
    actions = (np.random.random((1000, 4)), np.random.random((1000, 1)))
    rewards = np.random.random(1000)
    terminals = np.zeros(1000)

    kiox = create_simple_kiox_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    assert kiox.episode_manager.get_total_step_size() == 1000
    assert kiox.transition_buffer.size() == 999

    batch = kiox.sample(32)
    assert batch.observations[0].shape == (32, 100)
    assert batch.observations[1].shape == (32, 3, 84, 84)
    assert batch.actions[0].shape == (32, 4)
    assert batch.actions[1].shape == (32, 1)
    assert batch.next_observations[0].shape == (32, 100)
    assert batch.next_observations[1].shape == (32, 3, 84, 84)


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

    assert kiox.episode_manager.get_total_step_size() == 1000
    assert kiox.transition_buffer.size() == 999

    batch = kiox.sample(32)
    assert batch.observations.shape == (32, 3, 84, 84)
    assert batch.next_observations.shape == (32, 3, 84, 84)
