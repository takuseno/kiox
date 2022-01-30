from .kiox import Kiox
from .step_buffer import FIFOStepBuffer
from .transition_buffer import FIFOTransitionBuffer
from .transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
)


def create_simple_kiox(maxlen: int) -> Kiox:
    step_buffer = FIFOStepBuffer(maxlen)
    transition_buffer = FIFOTransitionBuffer(maxlen)
    return Kiox(step_buffer, transition_buffer, SimpleTransitionFactory())


def create_frame_stack_kiox(maxlen: int, n_frames: int) -> Kiox:
    step_buffer = FIFOStepBuffer(maxlen + n_frames)
    transition_buffer = FIFOTransitionBuffer(maxlen)
    transition_factory = FrameStackTransitionFactory(n_frames)
    return Kiox(step_buffer, transition_buffer, transition_factory)
