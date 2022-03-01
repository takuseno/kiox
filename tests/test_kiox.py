import io

import numpy as np

from kiox.kiox import Kiox
from kiox.transition_buffer import UnlimitedTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def test_kiox():
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox = Kiox(transition_buffer, transition_factory)

    for i in range(10):
        observation = np.random.random(100)
        action = np.random.random(4)
        reward = np.random.random()
        kiox.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=0.0,
        )
        assert kiox.get_transition_buffer_size() == i
        assert kiox.get_step_buffer_size() == i + 1

    # test sample
    batch = kiox.sample(5)
    assert batch.observations.shape == (5, 100)
    assert batch.actions.shape == (5, 4)
    assert batch.rewards.shape == (5, 1)
    assert batch.terminals.shape == (5, 1)

    # test copy_from
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox2 = Kiox(transition_buffer, transition_factory)
    kiox2.copy_from(kiox)
    assert kiox2.episode_manager.get_total_step_size() == 10
    assert kiox2.transition_buffer.size() == 9

    # test save
    io_byte = io.BytesIO()
    kiox.save(io_byte)

    # test load
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox3 = Kiox(transition_buffer, transition_factory)
    kiox3.load(io_byte)
    assert kiox3.episode_manager.get_total_step_size() == 10
