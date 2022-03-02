import numpy as np

from kiox.item import (
    locate_stacked_item,
    sizeof_stacked_item,
    stack_items,
    zeros_like,
)


def test_stack_items_float():
    items = [float(i) for i in range(10)]
    stacked_items = stack_items(items)
    assert stacked_items.shape == (10, 1)


def test_stack_items_ndarray():
    items = [np.random.random((3, 84, 84)) for _ in range(10)]
    stacked_items = stack_items(items)
    assert stacked_items.shape == (10, 3, 84, 84)


def test_stack_items_tuple():
    items = [
        (np.random.random(100), np.random.random((3, 84, 84)))
        for _ in range(10)
    ]
    stacked_items = stack_items(items)
    assert stacked_items[0].shape == (10, 100)
    assert stacked_items[1].shape == (10, 3, 84, 84)


def test_zeros_like_float():
    item = float(np.random.random())
    zero = zeros_like(item)
    assert zero == 0.0


def test_zeros_like_ndarray():
    item = np.random.random((3, 84, 84))
    zero = zeros_like(item)
    assert zero.shape == (3, 84, 84)
    assert np.all(zero == 0.0)


def test_zeros_like_tuple():
    item = (np.random.random(100), np.random.random((3, 84, 84)))
    zero = zeros_like(item)
    assert len(zero) == 2
    assert zero[0].shape == (100,)
    assert zero[1].shape == (3, 84, 84)
    assert np.all(zero[0] == 0.0)
    assert np.all(zero[1] == 0.0)


def test_sizeof_stacked_item_ndarray():
    stacked_item = np.random.random((10, 3, 84, 84))
    assert sizeof_stacked_item(stacked_item) == 10


def test_sizeof_stacked_item_tuple():
    stacked_item = (
        np.random.random((10, 100)),
        np.random.random((10, 3, 84, 84)),
    )
    assert sizeof_stacked_item(stacked_item) == 10


def test_locate_stacked_item_ndarray():
    stacked_item = np.random.random((10, 3, 84, 84))
    assert np.all(locate_stacked_item(stacked_item, 0) == stacked_item[0])


def test_locate_stacked_item_tuple():
    stacked_item = (
        np.random.random((10, 100)),
        np.random.random((10, 3, 84, 84)),
    )
    ref = (stacked_item[0][0], stacked_item[1][0])
    loc = locate_stacked_item(stacked_item, 0)
    assert np.all(loc[0] == ref[0])
    assert np.all(loc[1] == ref[1])
