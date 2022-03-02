from typing import Optional

from typing_extensions import Protocol

from .episode import Episode
from .step import Step
from .transition import (
    FrameStackLazyTransition,
    LazyTransition,
    SimpleLazyTransition,
)


class TransitionFactory(Protocol):
    """TransitionFactory object."""

    def create(
        self,
        step: Step,
        next_step: Optional[Step],
        episode: Episode,
        duration: int,
        gamma: float,
    ) -> LazyTransition:
        """Creates LazyTransition.

        Args:
            step: Step object.
            next_step: Step object at next step. If ``None``, ``step`` is
                terminal state.
            episode: Episode object including ``step`` and ``next_step``.
            duration: the number of steps before ``next_step``.
            gamma: discounted factor.

        Returns:
            LazyTransition object.

        """
        raise NotImplementedError


class SimpleTransitionFactory(TransitionFactory):
    """SimpleTransitionFactory class.

    This class creates SimpleLazyTransition.

    """

    def create(
        self,
        step: Step,
        next_step: Optional[Step],
        episode: Episode,
        duration: int,
        gamma: float,
    ) -> SimpleLazyTransition:
        return SimpleLazyTransition(
            curr_idx=step.idx,
            next_idx=None if next_step is None else next_step.idx,
            multi_step_reward=episode.compute_return(step.idx, duration, gamma),
            duration=duration,
        )


class FrameStackTransitionFactory(TransitionFactory):
    """FrameStackTransitionFactory class.

    This class creates FrameStackLazyTransition.

    Args:
        n_frames: number of frames to stack.

    """

    _n_frames: int

    def __init__(self, n_frames: int):
        self._n_frames = n_frames

    def create(
        self,
        step: Step,
        next_step: Optional[Step],
        episode: Episode,
        duration: int,
        gamma: float,
    ) -> FrameStackLazyTransition:
        prev_frames = []
        for i in range(self._n_frames - 1):
            prev_step = episode.get_prev(step.idx, i + 1)
            if prev_step is None:
                break
            prev_frames.append(prev_step.idx)
        return FrameStackLazyTransition(
            curr_idx=step.idx,
            next_idx=None if next_step is None else next_step.idx,
            multi_step_reward=episode.compute_return(step.idx, duration, gamma),
            duration=duration,
            prev_frames=list(reversed(prev_frames)),
            n_frames=self._n_frames,
        )
