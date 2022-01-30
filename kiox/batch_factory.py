import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence, Union, cast

import numpy as np

from .step_buffer import StepBuffer
from .transition_buffer import TransitionBuffer
from .types import Observation


@dataclasses.dataclass(frozen=True)
class Batch:
    observations: Union[np.ndarray, Sequence[np.ndarray]]
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: Union[np.ndarray, Sequence[np.ndarray]]
    terminals: np.ndarray
    durations: np.ndarray


def _stack_observations(
    observations: Sequence[Observation],
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    if isinstance(observations[0], np.ndarray):
        return np.asarray(
            np.stack(cast(Sequence[np.ndarray], observations), axis=0),
            dtype=np.float32,
        )
    return [
        np.asarray(
            np.stack(
                [observations[j][i] for j in range(len(observations))], axis=0
            ),
            dtype=np.float32,
        )
        for i in range(len(observations[0]))
    ]


class BatchFactory:
    _step_buffer: StepBuffer
    _transition_buffer: TransitionBuffer
    _max_pararellism: Optional[int]

    def __init__(
        self,
        step_buffer: StepBuffer,
        transition_buffer: TransitionBuffer,
        max_pararellism: Optional[int] = None,
    ):
        self._step_buffer = step_buffer
        self._transition_buffer = transition_buffer
        self._max_pararellism = max_pararellism

    def sample(self, batch_size: int) -> Batch:
        # multithreading could speed up I/O bounded codes
        with ThreadPoolExecutor(max_workers=self._max_pararellism) as executor:
            futures = [
                executor.submit(
                    self._transition_buffer.sample, self._step_buffer
                )
                for _ in range(batch_size)
            ]
            transitions = [future.result() for future in as_completed(futures)]

        # stack sampled data
        observations = _stack_observations(
            [transition.observation for transition in transitions]
        )
        next_observations = _stack_observations(
            [transition.next_observation for transition in transitions]
        )
        actions = np.array(
            [transition.action for transition in transitions], dtype=np.float32
        )
        rewards = np.array(
            [transition.reward for transition in transitions], dtype=np.float32
        )
        terminals = np.array(
            [transition.terminal for transition in transitions],
            dtype=np.float32,
        )
        durations = np.array(
            [transition.duration for transition in transitions],
            dtype=np.float32,
        )

        return Batch(
            observations=observations,
            actions=np.reshape(actions, [batch_size, -1]),
            rewards=np.reshape(rewards, [batch_size, -1]),
            next_observations=next_observations,
            terminals=np.reshape(terminals, [batch_size, -1]),
            durations=np.reshape(durations, [batch_size, -1]),
        )
