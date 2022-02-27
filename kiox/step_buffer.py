import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np
from typing_extensions import Protocol

from .item import Item


@dataclasses.dataclass(frozen=True)
class Step:
    idx: int
    observation: Item
    action: Item
    reward: Item
    terminal: float


class StepBuffer(Protocol):
    def append(self, step: Step) -> None:
        raise NotImplementedError

    def get(self, idx: int) -> Step:
        raise NotImplementedError

    def get_by_index(self, index: int) -> Step:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def copy_from(self, step_buffer: "StepBuffer") -> None:
        raise NotImplementedError

    @property
    def steps(self) -> Sequence[Step]:
        raise NotImplementedError


class UnlimitedStepBuffer(StepBuffer):
    _buffer: List[Step]
    _index: Dict[int, int]

    def __init__(self) -> None:
        self._buffer = []
        self._index = {}

    def append(self, step: Step) -> None:
        self._buffer.append(step)
        self._index[len(self._buffer) - 1] = step.idx

    def size(self) -> int:
        return len(self._buffer)

    def get(self, idx: int) -> Step:
        assert idx in self._index, f"Step(idx={idx}) does not exist"
        return self._buffer[self._index[idx]]

    def get_by_index(self, index: int) -> Step:
        return self._buffer[index]

    def clear(self) -> None:
        self._buffer.clear()
        self._index = {}

    def copy_from(self, step_buffer: StepBuffer) -> None:
        for i in range(step_buffer.size()):
            self.append(step_buffer.get_by_index(i))

    @property
    def steps(self) -> Sequence[Step]:
        return self._buffer


class FIFOStepBuffer(StepBuffer):
    _maxlen: int
    _buffer: List[Step]
    _index: Dict[int, int]
    _cursor: int

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._buffer = []
        self._index = {}
        self._cursor = 0

    def append(self, step: Step) -> None:
        if len(self._buffer) < self._maxlen:
            self._buffer.append(step)
        else:
            del self._index[self._buffer[self._cursor].idx]
            self._buffer[self._cursor] = step

        self._index[step.idx] = self._cursor

        self._cursor += 1
        if self._cursor == self._maxlen:
            self._cursor = 0

    def get(self, idx: int) -> Step:
        assert idx in self._index, f"Step(idx={idx}) does not exist"
        return self._buffer[self._index[idx]]

    def get_by_index(self, index: int) -> Step:
        return self._buffer[index]

    def size(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._index = {}
        self._cursor = 0

    def copy_from(self, step_buffer: StepBuffer) -> None:
        for i in range(step_buffer.size()):
            self.append(step_buffer.get_by_index(i))

    @property
    def steps(self) -> Sequence[Step]:
        return self._buffer


class EpisodicStepBuffer(StepBuffer):
    _buffer: List[Step]
    _index: Dict[int, int]
    _prev_index: Dict[int, int]
    _next_index: Dict[int, int]
    _prev_step: Optional[Step]
    _cursor: int

    def __init__(self) -> None:
        self._buffer = []
        self._index = {}
        self._prev_index = {}
        self._next_index = {}
        self._prev_step = None
        self._cursor = 0

    def append(self, step: Step) -> None:
        self._buffer.append(step)
        self._index[step.idx] = self._cursor
        if self._prev_step:
            self._prev_index[step.idx] = self._index[self._prev_step.idx]
            self._next_index[self._prev_step.idx] = self._cursor
        self._cursor += 1
        self._prev_step = step

    def get(self, idx: int) -> Step:
        assert idx in self._index, f"Step(idx={idx}) does not exist"
        return self._buffer[self._index[idx]]

    def get_by_index(self, index: int) -> Step:
        return self._buffer[index]

    def get_next(self, idx: int, duration: int = 1) -> Optional[Step]:
        next_idx = idx
        for _ in range(duration):
            if next_idx not in self._next_index:
                return None
            next_idx = self._buffer[self._next_index[next_idx]].idx
        return self.get(next_idx)

    def get_prev(self, idx: int, duration: int = 1) -> Optional[Step]:
        prev_idx = idx
        for _ in range(duration):
            if prev_idx not in self._prev_index:
                return None
            prev_idx = self._buffer[self._prev_index[prev_idx]].idx
        return self.get(prev_idx)

    def compute_return(
        self, idx: int, duration: int = 1, gamma: float = 0.99
    ) -> float:
        next_idx = idx
        ret = 0.0
        for i in range(duration):
            reward = self.get(next_idx).reward
            assert isinstance(reward, (float, np.ndarray))
            ret += (gamma**i) * reward
            if next_idx in self._next_index:
                next_idx = self._next_index[next_idx]
            else:
                break
        return ret

    def size(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._index = {}
        self._prev_index = {}
        self._next_index = {}
        self._prev_step = None
        self._cursor = 0

    def copy_from(self, step_buffer: StepBuffer) -> None:
        raise NotImplementedError(
            "EpisodicStepBuffer does not support copy_from."
        )

    @property
    def steps(self) -> Sequence[Step]:
        return self._buffer
