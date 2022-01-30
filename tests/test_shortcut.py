from kiox.kiox import Kiox
from kiox.shortcut import create_frame_stack_kiox, create_simple_kiox
from kiox.step_buffer import FIFOStepBuffer
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
)


def test_create_simple_kiox():
    kiox = create_simple_kiox(10)
    assert isinstance(kiox, Kiox)
    assert isinstance(kiox.step_buffer, FIFOStepBuffer)
    assert isinstance(kiox.transition_buffer, FIFOTransitionBuffer)
    assert isinstance(kiox.transition_factory, SimpleTransitionFactory)


def test_create_frame_stack_kiox():
    kiox = create_frame_stack_kiox(10, 3)
    assert isinstance(kiox, Kiox)
    assert isinstance(kiox.step_buffer, FIFOStepBuffer)
    assert isinstance(kiox.transition_buffer, FIFOTransitionBuffer)
    assert isinstance(kiox.transition_factory, FrameStackTransitionFactory)
