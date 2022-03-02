from typing import Sequence, Union

import numpy as np

Item = Union[int, float, np.ndarray, Sequence[np.ndarray]]
StackedItem = Union[np.ndarray, Sequence[np.ndarray]]


def stack_items(items: Sequence[Item]) -> StackedItem:
    """Stacks a sequence of items.

    In case of ``int`` or ``float`` sequence:

    .. code-block:: python

        items = [1.0, 2.0, 3.0]
        stacked_item = stack_items(items)
        assert stacked_item.shape == (3, 1)

    In case of ``np.ndarray``:

    .. code-block:: python

        items = [np.random.random(4), np.random.random(4)]
        stacked_item = stacked_items(items)
        assert stacked_items.shape == (2, 4)

    In case of a list of sequences:

    .. code-block:: python

        items = [
            (np.random.random(2), np.random.random(4)),
            (np.random.random(2), np.random.random(4)),
            (np.random.random(2), np.random.random(4)),
        ]
        stacked_item = stacked_items(items)
        assert stacked_items[0].shape == (3, 2)
        assert stacked_items[1].shape == (3, 4)

    Args:
        items: a list of items.

    Returns:
        stacked items.

    """
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
    """Creates identically shaped item filled with zeros.

    Args:
        item: item.

    Returns:
        item filled with zeros.

    """
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


def sizeof_stacked_item(stacked_item: StackedItem) -> int:
    """Returns size of stacked item.

    Args:
        stacked_item: stacked items.

    Returns:
        size of sequence.

    """
    if isinstance(stacked_item, (list, tuple)):
        size = len(stacked_item[0])
        for item in stacked_item[1:]:
            assert item.shape[0] == size, "all elements must have same size."
        return size
    else:
        assert isinstance(stacked_item, np.ndarray)
        return int(stacked_item.shape[0])


def locate_stacked_item(stacked_item: StackedItem, index: int) -> Item:
    """Returns item located by ``index``.

    Args:
        stacked_item: stacked items.
        index: location.

    Returns:
        located item.

    """
    if isinstance(stacked_item, (list, tuple)):
        return [item[index] for item in stacked_item]
    else:
        assert isinstance(stacked_item, np.ndarray)
        return stacked_item[index]
