import dataclasses
from typing import Optional, Sequence, cast

import numpy as np

from .item import Item, zeros_like
from .step import StepBuffer


@dataclasses.dataclass(frozen=True)
class Transition:
    observation: Item
    action: Item
    reward: Item
    next_observation: Item
    terminal: Item
    duration: int


@dataclasses.dataclass(frozen=True)
class LazyTransition:
    curr_idx: int
    next_idx: Optional[int]
    multi_step_reward: float
    duration: int

    def create(self, step_buffer: StepBuffer) -> Transition:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class SimpleLazyTransition(LazyTransition):
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
