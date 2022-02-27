# pylint: disable=R1711
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np
from typing_extensions import Protocol

from .episode import EpisodeManager
from .transition import LazyTransition, Transition


class TransitionBuffer(Protocol):
    def append(
        self, lazy_transition: LazyTransition
    ) -> Optional[LazyTransition]:
        raise NotImplementedError

    def get_by_index(self, index: int) -> LazyTransition:
        raise NotImplementedError

    def sample(self, episode_manager: EpisodeManager) -> Transition:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def copy_from(self, transition_buffer: "TransitionBuffer") -> None:
        raise NotImplementedError

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        raise NotImplementedError


class UnlimitedTransitionBuffer(TransitionBuffer):
    _buffer: List[LazyTransition]

    def __init__(self) -> None:
        self._buffer = []

    def append(
        self, lazy_transition: LazyTransition
    ) -> Optional[LazyTransition]:
        self._buffer.append(lazy_transition)
        return None

    def get_by_index(self, index: int) -> LazyTransition:
        return self._buffer[index]

    def sample(self, episode_manager: EpisodeManager) -> Transition:
        index = int(np.random.randint(len(self._buffer)))
        return self._buffer[index].create(episode_manager)

    def size(self) -> int:
        return len(self._buffer)

    def copy_from(self, transition_buffer: TransitionBuffer) -> None:
        for i in range(transition_buffer.size()):
            self.append(transition_buffer.get_by_index(i))

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        return self._buffer


class FIFOTransitionBuffer(TransitionBuffer):
    _maxlen: int
    _buffer: Deque[LazyTransition]

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)

    def append(
        self, lazy_transition: LazyTransition
    ) -> Optional[LazyTransition]:
        dropped_transition: Optional[LazyTransition]
        if self.size() == self._maxlen:
            dropped_transition = self._buffer[0]
        else:
            dropped_transition = None
        self._buffer.append(lazy_transition)
        return dropped_transition

    def get_by_index(self, index: int) -> LazyTransition:
        return self._buffer[index]

    def sample(self, episode_manager: EpisodeManager) -> Transition:
        index = int(np.random.randint(len(self._buffer)))
        return self._buffer[index].create(episode_manager)

    def size(self) -> int:
        return len(self._buffer)

    def copy_from(self, transition_buffer: TransitionBuffer) -> None:
        for i in range(transition_buffer.size()):
            self.append(transition_buffer.get_by_index(i))

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        return self._buffer
