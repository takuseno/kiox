import dataclasses
from typing import Dict, Sequence

from .item import Item


@dataclasses.dataclass(frozen=True)
class Step:
    idx: int
    observation: Item
    action: Item
    reward: Item
    terminal: float


class StepBuffer:
    _steps: Dict[int, Step]

    def __init__(self) -> None:
        self._steps = {}

    def get(self, idx: int) -> Step:
        assert idx in self._steps, f"Step(idx={idx}) does not exist"
        return self._steps[idx]

    def append(self, step: Step) -> None:
        assert step.idx not in self._steps
        self._steps[step.idx] = step

    def drop(self, idx: int) -> None:
        del self._steps[idx]

    def size(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> Sequence[Step]:
        return list(self._steps.values())
