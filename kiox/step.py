import dataclasses
from typing import Dict, Sequence

from .item import Item


@dataclasses.dataclass(frozen=True)
class Step:
    """Step data class.

    Args:
        idx: unique id to identify steps.
        observation: observation.
        action: action.
        reward: reward.
        terminal: terminal flag.

    """

    idx: int
    observation: Item
    action: Item
    reward: Item
    terminal: float


class StepBuffer:
    """StepBuffer class."""

    _steps: Dict[int, Step]

    def __init__(self) -> None:
        self._steps = {}

    def get(self, idx: int) -> Step:
        """Returns step by specified ``idx``.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        assert idx in self._steps, f"Step(idx={idx}) does not exist"
        return self._steps[idx]

    def append(self, step: Step) -> None:
        """Appends step.

        Args:
            step: Step object.

        """
        assert step.idx not in self._steps
        self._steps[step.idx] = step

    def drop(self, idx: int) -> None:
        """Drops step by specified ``idx``.

        Args:
            idx: step idx.

        """
        del self._steps[idx]

    def size(self) -> int:
        """Returns number of stored steps.

        Returns:
            number of stored steps.

        """
        return len(self._steps)

    @property
    def steps(self) -> Sequence[Step]:
        return list(self._steps.values())
