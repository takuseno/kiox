import io

import numpy as np

from kiox.kiox import Kiox
from kiox.step_buffer import UnlimitedStepBuffer
from kiox.transition_buffer import UnlimitedTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def test_kiox():
    step_buffer = UnlimitedStepBuffer()
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox = Kiox(step_buffer, transition_buffer, transition_factory)

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
        assert kiox.step_buffer.size() == i + 1
        assert kiox.transition_buffer.size() == i

    # test sample
    batch = kiox.sample(5)
    assert batch.observations.shape == (5, 100)
    assert batch.actions.shape == (5, 4)
    assert batch.rewards.shape == (5, 1)
    assert batch.terminals.shape == (5, 1)

    # test copy_from
    step_buffer = UnlimitedStepBuffer()
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox2 = Kiox(step_buffer, transition_buffer, transition_factory)
    kiox2.copy_from(kiox)
    assert kiox2.step_buffer.size() == 10
    assert kiox2.transition_buffer.size() == 9

    # test save
    io_byte = io.BytesIO()
    kiox.save(io_byte)

    # test load
    step_buffer = UnlimitedStepBuffer()
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    kiox3 = Kiox(step_buffer, transition_buffer, transition_factory)
    kiox3.load(io_byte)
    assert kiox3.step_buffer.size() == 10
