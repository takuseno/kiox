from collections import deque
from typing import IO, Deque, List, Sequence

import numpy as np
from typing_extensions import Protocol

from .step_buffer import StepBuffer
from .transition import LazyTransition, Transition


class TransitionBuffer(Protocol):
    def append(self, lazy_transition: LazyTransition) -> None:
        raise NotImplementedError

    def get_by_index(self, index: int) -> LazyTransition:
        raise NotImplementedError

    def sample(self, step_buffer: StepBuffer) -> Transition:
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

    def __init__(self):
        self._buffer = []

    def append(self, lazy_transition: LazyTransition) -> None:
        self._buffer.append(lazy_transition)

    def get_by_index(self, index: int) -> LazyTransition:
        return self._buffer[index]

    def sample(self, step_buffer: StepBuffer) -> Transition:
        index = np.random.randint(len(self._buffer))
        return self._buffer[index].create(step_buffer)

    def size(self) -> int:
        return len(self._buffer)

    def copy_from(self, transition_buffer: TransitionBuffer) -> None:
        for i in range(transition_buffer.size()):
            self.append(transition_buffer.get_by_index(i))

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        return self._buffer


class FIFOTransitionBuffer(TransitionBuffer):
    _buffer: Deque[LazyTransition]

    def __init__(self, maxlen: int):
        self._buffer = deque(maxlen=maxlen)

    def append(self, lazy_transition: LazyTransition) -> None:
        self._buffer.append(lazy_transition)

    def get_by_index(self, index: int) -> LazyTransition:
        return self._buffer[index]

    def sample(self, step_buffer: StepBuffer) -> Transition:
        index = np.random.randint(len(self._buffer))
        return self._buffer[index].create(step_buffer)

    def size(self) -> int:
        return len(self._buffer)

    def copy_from(self, transition_buffer: TransitionBuffer) -> None:
        for i in range(transition_buffer.size()):
            self.append(transition_buffer.get_by_index(i))

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        return self._buffer
