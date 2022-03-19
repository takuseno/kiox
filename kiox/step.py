import dataclasses
from typing import Dict, Sequence

from .item import Item


@dataclasses.dataclass(frozen=True)
class PartialStep:
    """Step data class without idx.

    Args:
        observation: observation.
        action: action.
        reward: reward.
        terminal: terminal flag.

    """

    observation: Item
    action: Item
    reward: Item
    terminal: float


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

    def to_partial_step(self) -> PartialStep:
        return PartialStep(
            observation=self.observation,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
        )


class StepBuffer:
    """StepBuffer class."""

    _steps: Dict[int, Step]

    def __init__(self) -> None:
        self._steps = {}
        self._counter = 0

    def get(self, idx: int) -> Step:
        """Returns step by specified ``idx``.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        assert idx in self._steps, f"Step(idx={idx}) does not exist"
        return self._steps[idx]

    def append(self, partial_step: PartialStep) -> Step:
        """Appends step.

        Args:
            partial_step: PartialStep object.

        Returns:
            Step object.

        """
        idx = self._counter
        step = Step(
            idx=idx,
            observation=partial_step.observation,
            action=partial_step.action,
            reward=partial_step.reward,
            terminal=partial_step.terminal,
        )
        self._steps[idx] = step
        self._counter += 1
        return self._steps[idx]

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
