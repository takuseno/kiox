from kiox.transition import Transition
from kiox.transition_buffer import (
    FIFOTransitionBuffer,
    UnlimitedTransitionBuffer,
)

from .utility import StepFactory, TransitionFactory


def test_unlimited_transition_buffer():
    factory = TransitionFactory(StepFactory())
    buffer = UnlimitedTransitionBuffer()
    transitions = []

    for i in range(10):
        transition = factory()
        buffer.append(transition)
        transitions.append(transition)
        assert buffer.size() == i + 1

    assert buffer.get_by_index(0) is transitions[0]

    # test sample
    transition = buffer.sample(factory.step_buffer)
    assert isinstance(transition, Transition)


def test_fifo_transition_buffer():
    factory = TransitionFactory(StepFactory())
    buffer = FIFOTransitionBuffer(5)
    transitions = []

    for i in range(10):
        transition = factory()
        buffer.append(transition)
        transitions.append(transition)
        assert buffer.size() == min(i + 1, 5)

    assert buffer.get_by_index(0) is transitions[5]

    # test sample
    transition = buffer.sample(factory.step_buffer)
    assert isinstance(transition, Transition)
