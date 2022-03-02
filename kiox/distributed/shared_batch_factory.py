# pylint: disable=C0200
from typing import Optional, Sequence, Union

import numpy as np

from ..batch_factory import Batch, BatchFactory
from ..step import StepBuffer
from ..transition_buffer import TransitionBuffer
from .shared_array import create_shared_array


def _create_shared_batch_array(
    batch_size: int, shape: Union[Sequence[Sequence[int]], Sequence[int]]
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    if isinstance(shape[0], int):
        return create_shared_array((batch_size, *shape), np.float32)
    else:
        return [
            create_shared_array((batch_size, *s), np.float32) for s in shape  # type: ignore
        ]


def _copy_array(
    dst: Union[Sequence[np.ndarray], np.ndarray],
    src: Union[Sequence[np.ndarray], np.ndarray],
) -> None:
    if isinstance(dst, (list, tuple)):
        for i in range(len(dst)):
            np.copyto(dst[i], src[i])
    else:
        np.copyto(dst, src)


class SharedBatchFactory:
    """SharedBatchFactory class.

    Args:
        observation_shape: shape of observation.
        action_shape: shape of action.
        reward_shape: shape of reward.
        batch_size: batch size.

    """

    _batch_size: int
    _batch: Batch

    def __init__(
        self,
        observation_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        action_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        reward_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        batch_size: int,
    ):
        self._batch_size = batch_size

        # allocate shared arrays
        observations = _create_shared_batch_array(batch_size, observation_shape)
        actions = _create_shared_batch_array(batch_size, action_shape)
        rewards = _create_shared_batch_array(batch_size, reward_shape)
        terminals = create_shared_array((batch_size, 1), np.float32)
        next_observations = _create_shared_batch_array(
            batch_size, observation_shape
        )
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
        """Samples transitions and copies mini-batch to shared memory.

        Args:
            step_buffer: StepBuffer object.
            transition_buffer: TransitionBuffer object.
            max_pararellism: maximum number of threads to sample.

        """
        # sampling
        factory = BatchFactory(
            step_buffer=step_buffer,
            transition_buffer=transition_buffer,
            max_pararellism=max_pararellism,
        )
        batch = factory.sample(self._batch_size)

        # copy arrays
        _copy_array(self._batch.observations, batch.observations)
        _copy_array(self._batch.actions, batch.actions)
        _copy_array(self._batch.rewards, batch.rewards)
        np.copyto(self._batch.terminals, batch.terminals)
        _copy_array(self._batch.next_observations, batch.next_observations)
        np.copyto(self._batch.durations, batch.durations)

    @property
    def batch(self) -> Batch:
        return self._batch
