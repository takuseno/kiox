from .kiox import Kiox
from .transition_buffer import FIFOTransitionBuffer
from .transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
)


def create_simple_kiox(
    maxlen: int, n_steps: int = 1, gamma: float = 0.99
) -> Kiox:
    """Alias of Kiox with FIFOTransitionBuffer and SimpleTransitionFactory.

    Args:
        maxlen: maximum size of transitions.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    Returns:
        Kiox object.

    """
    transition_buffer = FIFOTransitionBuffer(maxlen)
    return Kiox(
        transition_buffer,
        SimpleTransitionFactory(),
        n_steps=n_steps,
        gamma=gamma,
    )


def create_frame_stack_kiox(
    maxlen: int, n_frames: int, n_steps: int = 1, gamma: float = 0.99
) -> Kiox:
    """Alias of Kiox with FIFOTransitionBuffer and FrameStackTransitionFactory.

    Args:
        maxlen: maximum size of transitions.
        n_frames: number of frames to stack.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    Returns:
        Kiox object.

    """
    transition_buffer = FIFOTransitionBuffer(maxlen)
    transition_factory = FrameStackTransitionFactory(n_frames)
    return Kiox(
        transition_buffer,
        transition_factory,
        n_steps=n_steps,
        gamma=gamma,
    )
