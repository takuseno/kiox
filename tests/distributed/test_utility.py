import numpy as np
import pytest

from kiox.distributed.proto.step_pb2 import (
    ContinuousAction,
    DiscreteAction,
    DType,
)
from kiox.distributed.utility import (
    convert_action_to_proto,
    convert_dtype_to_proto,
    convert_observation_to_proto,
    convert_proto_to_action,
    convert_proto_to_dtype,
    convert_proto_to_observation,
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


def test_convert_observation_to_proto():
    observation = np.random.random((3, 84, 84)).astype(np.float32)
    proto = convert_observation_to_proto(observation)
    array = np.frombuffer(proto.data[0], dtype=np.float32)
    assert proto.length == 1
    assert proto.dtype[0] == DType.FLOAT32
    assert proto.shape[0].dim == [3, 84, 84]
    assert np.all(array == observation.reshape(-1))


def test_convert_proto_to_observation():
    observation = np.random.random((3, 84, 84)).astype(np.float32)
    proto = convert_observation_to_proto(observation)
    converted_observation = convert_proto_to_observation(proto)
    assert converted_observation.shape == observation.shape
    assert np.all(converted_observation == observation)


@pytest.mark.parametrize("action", [np.random.random(4), 3])
def test_convert_action_to_proto(action):
    proto = convert_action_to_proto(action)
    if isinstance(action, np.ndarray):
        assert isinstance(proto, ContinuousAction)
        assert proto.length == action.size
        assert np.allclose(proto.data, action.tolist())
    else:
        assert isinstance(proto, DiscreteAction)
        assert proto.data == action


@pytest.mark.parametrize("action", [np.random.random(4), 3])
def test_convert_proto_to_action(action):
    proto = convert_action_to_proto(action)
    converted_action = convert_proto_to_action(proto)
    if isinstance(action, np.ndarray):
        assert isinstance(converted_action, np.ndarray)
        assert np.allclose(converted_action, action)
    else:
        assert isinstance(converted_action, int)
        assert converted_action == action
