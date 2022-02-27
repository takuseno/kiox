import dataclasses

from .item import Item


@dataclasses.dataclass(frozen=True)
class Step:
    idx: int
    observation: Item
    action: Item
    reward: Item
    terminal: float
