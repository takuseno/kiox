# pylint: disable=C0200
from typing import Optional, Sequence, Union

import numpy as np

from ..batch_factory import Batch, BatchFactory
from ..step_buffer import StepBuffer
from ..transition_buffer import TransitionBuffer
from .shared_array import create_shared_array


def _create_shared_observation(
    batch_size: int, shape: Union[Sequence[Sequence[int]], Sequence[int]]
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    if isinstance(shape[0], int):
        return create_shared_array((batch_size, *shape), np.float32)
    else:
        return [
            create_shared_array((batch_size, *s), np.float32) for s in shape  # type: ignore
        ]


class SharedBatchFactory:
    _batch_size: int
    _observation_shape: Union[Sequence[Sequence[int]], Sequence[int]]
    _action_shape: Sequence[int]
    _batch: Batch

    def __init__(
        self,
        observation_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        action_shape: Sequence[int],
        batch_size: int,
    ):
        self._batch_size = batch_size
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        # allocate shared arrays
        observations = _create_shared_observation(batch_size, observation_shape)
        actions = create_shared_array((batch_size, *action_shape), np.float32)
        rewards = create_shared_array((batch_size, 1), np.float32)
        next_observations = _create_shared_observation(
            batch_size, observation_shape
        )
        terminals = create_shared_array((batch_size, 1), np.float32)
        durations = create_shared_array((batch_size, 1), np.float32)
        self._batch = Batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            durations=durations,
        )

    def sample(
        self,
        step_buffer: StepBuffer,
        transition_buffer: TransitionBuffer,
        max_pararellism: Optional[int] = None,
    ) -> None:
        # sampling
        factory = BatchFactory(step_buffer, transition_buffer, max_pararellism)
        batch = factory.sample(self._batch_size)

        # copy arrays
        if isinstance(self._batch.observations, list):
            for i in range(len(self._batch.observations)):
                np.copyto(self._batch.observations[i], batch.observations[i])
                np.copyto(
                    self._batch.next_observations[i], batch.next_observations[i]
                )
        else:
            np.copyto(self._batch.observations, batch.observations)
            np.copyto(self._batch.next_observations, batch.next_observations)
        np.copyto(self._batch.actions, batch.actions)
        np.copyto(self._batch.rewards, batch.rewards)
        np.copyto(self._batch.terminals, batch.terminals)
        np.copyto(self._batch.durations, batch.durations)

    @property
    def batch(self) -> Batch:
        return self._batch
