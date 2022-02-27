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
    def create(
        self,
        step: Step,
        next_step: Optional[Step],
        episode: Episode,
        duration: int,
        gamma: float,
    ) -> LazyTransition:
        raise NotImplementedError


class SimpleTransitionFactory(TransitionFactory):
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
