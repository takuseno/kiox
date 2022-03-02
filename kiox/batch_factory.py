import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from .item import StackedItem, stack_items
from .step import StepBuffer
from .transition_buffer import TransitionBuffer


@dataclasses.dataclass(frozen=True)
class Batch:
    """Mini-batch data class.

    Args:
        observations: observation batch.
        actions: action batch.
        rewards: reward batch.
        next_observations: next observation batch.
        terminals: terminal flag batch.
        durations: step duration batch.

    """

    observations: StackedItem
    actions: StackedItem
    rewards: StackedItem
    next_observations: StackedItem
    terminals: np.ndarray
    durations: np.ndarray


class BatchFactory:
    """BatchFactory class.

    This class samples mini-batch by internally evaluating LazyTransition
    objects. The batch creation is multi-threaded so that I/O bounded
    transition evaluation might be faster.

    Args:
        step_buffer: StepBuffer object.
        transition_buffer: TransitionBuffer object.
        max_pararellism: maximum number of threads.

    """

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
        """Samples transitions and returns mini-batch.

        Args:
            batch_size: batch size.

        Returns:
            mini-batch.

        """
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
        observations = stack_items(
            [transition.observation for transition in transitions]
        )
        next_observations = stack_items(
            [transition.next_observation for transition in transitions]
        )
        actions = stack_items([transition.action for transition in transitions])
        rewards = stack_items([transition.reward for transition in transitions])
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
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=np.reshape(terminals, [batch_size, -1]),
            durations=np.reshape(durations, [batch_size, -1]),
        )
