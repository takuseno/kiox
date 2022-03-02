import dataclasses
from typing import Optional, Sequence, cast

import numpy as np

from .item import Item, zeros_like
from .step import StepBuffer


@dataclasses.dataclass(frozen=True)
class Transition:
    """Transtion data class.

    Args:
        observation: observation.
        action: action.
        reward: reward.
        next_observation: next observation.
        terminal: terminal flag.
        duration: number of steps before next observation.

    """

    observation: Item
    action: Item
    reward: Item
    next_observation: Item
    terminal: Item
    duration: int


@dataclasses.dataclass(frozen=True)
class LazyTransition:
    """LazyTransition class.

    This class does not have step data itself. Instead, this class has idx
    pointing to Step object and lazily generates Transition object.

    Args:
        curr_idx: idx for the current step.
        next_idx: idx for the next step. If ``None``, next step is terminal
            state.
        multi_step_reward: discounted return during this transition.
        duration: the number of steps before next step.

    """

    curr_idx: int
    next_idx: Optional[int]
    multi_step_reward: float
    duration: int

    def create(self, step_buffer: StepBuffer) -> Transition:
        """Builds Transition.

        Args:
            step_buffer: StepBuffer object.

        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class SimpleLazyTransition(LazyTransition):
    """SimpleLazyTransition class.

    This class is the most plain lazy transition.

    Args:
        curr_idx: idx for the current step.
        next_idx: idx for the next step. If ``None``, next step is terminal
            state.
        multi_step_reward: discounted return during this transition.
        duration: the number of steps before next step.

    """

    def create(self, step_buffer: StepBuffer) -> Transition:
        step = step_buffer.get(self.curr_idx)
        observation = step.observation
        if self.next_idx is None:
            next_observation = zeros_like(observation)
        else:
            next_step = step_buffer.get(self.next_idx)
            next_observation = next_step.observation
        return Transition(
            observation=observation,
            action=step.action,
            reward=self.multi_step_reward,
            next_observation=next_observation,
            terminal=step.terminal,
            duration=self.duration,
        )


@dataclasses.dataclass(frozen=True)
class FrameStackLazyTransition(LazyTransition):
    """FrameStackLazyTransition class.

    This class is designed to stack recent frames at transition creation.

    Args:
        curr_idx: idx for the current step.
        next_idx: idx for the next step. If ``None``, next step is terminal
            state.
        multi_step_reward: discounted return during this transition.
        duration: the number of steps before next step.
        prev_frames: idx list of previous frames to stack.
        n_frames: number of frames to stack.

    """

    prev_frames: Sequence[int]
    n_frames: int

    def create(self, step_buffer: StepBuffer) -> Transition:
        step = step_buffer.get(self.curr_idx)
        observation = step.observation
        assert isinstance(observation, np.ndarray)

        # fill with padding
        n_pads = max(self.n_frames - len(self.prev_frames) - 1, 0)
        frames = [np.zeros_like(observation) for _ in range(n_pads)]

        # stack previous frames
        frames += [
            cast(np.ndarray, step_buffer.get(idx).observation)
            for idx in self.prev_frames
        ]

        # stack the latest frame
        frames += [observation]

        stacked_observation = np.vstack(frames)

        if self.next_idx is None:
            stacked_next_observation = np.zeros_like(stacked_observation)
        else:
            next_step = step_buffer.get(self.next_idx)
            next_observation = next_step.observation
            assert isinstance(next_observation, np.ndarray)
            next_frames = frames[1:] + [next_observation]
            stacked_next_observation = np.vstack(next_frames)

        return Transition(
            observation=stacked_observation,
            action=step.action,
            reward=self.multi_step_reward,
            next_observation=stacked_next_observation,
            terminal=step.terminal,
            duration=self.duration,
        )
