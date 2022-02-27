from typing import Sequence, Union

import numpy as np

Item = Union[int, float, np.ndarray, Sequence[np.ndarray]]
StackedItem = Union[np.ndarray, Sequence[np.ndarray]]


def stack_items(items: Sequence[Item]) -> StackedItem:
    item = items[0]
    if isinstance(item, (int, float)):
        stacked_items = np.reshape(np.array(items), [-1, 1])
    elif isinstance(item, np.ndarray):
        stacked_items = np.stack(items, axis=0)
    elif isinstance(item, (list, tuple)):
        assert isinstance(item[0], np.ndarray)
        stacked_items = [
            np.array([items[j][i] for j in range(len(items))])  # type: ignore
            for i in range(len(item))
        ]
    else:
        raise ValueError(f"unrecognized item type: {type(item)}")
    return stacked_items


def zeros_like(item: Item) -> Item:
    zeros: Item
    if isinstance(item, (int, float)):
        zeros = 0 if isinstance(item, int) else 0.0
    elif isinstance(item, np.ndarray):
        zeros = np.zeros_like(item)
    elif isinstance(item, (list, tuple)):
        zeros = [np.zeros_like(el) for el in item]
    else:
        raise ValueError(f"unrecognized item type: {type(item)}")
    return zeros
