import numpy as np
import pytest

from kiox.distributed.proto.step_pb2 import DType
from kiox.distributed.utility import (
    convert_dtype_to_proto,
    convert_item_to_proto,
    convert_proto_to_dtype,
    convert_proto_to_item,
)


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32])
def test_convert_dtype_to_proto(dtype):
    proto = convert_dtype_to_proto(dtype)
    if dtype == np.uint8:
        assert proto == DType.UINT8
    elif dtype == np.int32:
        assert proto == DType.INT32
    elif dtype == np.float32:
        assert proto == DType.FLOAT32


@pytest.mark.parametrize("proto", [DType.UINT8, DType.INT32, DType.FLOAT32])
def test_convert_proto_to_dtype(proto):
    dtype = convert_proto_to_dtype(proto)
    if proto == DType.UINT8:
        assert dtype == np.uint8
    elif proto == DType.INT32:
        assert dtype == np.int32
    elif proto == DType.FLOAT32:
        assert dtype == np.float32


@pytest.mark.parametrize(
    "item",
    [
        float(np.random.random()),
        np.random.random((3, 84, 84)).astype(np.float32),
        [
            np.random.random(100).astype(np.float32),
            np.random.random((3, 84, 84)).astype(np.float32),
        ],
    ],
)
def test_convert_item_to_proto(item):
    proto = convert_item_to_proto(item)
    if isinstance(item, float):
        value = np.frombuffer(proto.data[0], dtype=np.float32)
        assert proto.length == 1
        assert proto.dtype[0] == DType.FLOAT32
        assert proto.shape[0].dim == [1]
        assert value == item
    elif isinstance(item, np.ndarray):
        value = np.frombuffer(proto.data[0], dtype=np.float32)
        assert proto.length == 1
        assert proto.dtype[0] == DType.FLOAT32
        assert proto.shape[0].dim == [3, 84, 84]
        assert np.all(value == item.reshape(-1))
    else:
        value1 = np.frombuffer(proto.data[0], dtype=np.float32)
        value2 = np.frombuffer(proto.data[1], dtype=np.float32)
        assert proto.length == 2
        assert proto.dtype[0] == DType.FLOAT32
        assert proto.dtype[1] == DType.FLOAT32
        assert proto.shape[0].dim == [100]
        assert proto.shape[1].dim == [3, 84, 84]
        assert np.all(value1 == item[0].reshape(-1))
        assert np.all(value2 == item[1].reshape(-1))


@pytest.mark.parametrize(
    "item",
    [
        float(np.random.random()),
        np.random.random((3, 84, 84)).astype(np.float32),
        [
            np.random.random(100).astype(np.float32),
            np.random.random((3, 84, 84)).astype(np.float32),
        ],
    ],
)
def test_convert_proto_to_item(item):
    proto = convert_item_to_proto(item)
    converted_item = convert_proto_to_item(proto)
    if isinstance(item, float):
        assert converted_item.shape == (1,)
        assert np.allclose(converted_item[0], item)
    elif isinstance(item, np.ndarray):
        assert converted_item.shape == item.shape
        assert np.all(converted_item == item)
    else:
        for i in range(2):
            assert converted_item[i].shape == item[i].shape
            assert np.all(converted_item[i] == item[i])
