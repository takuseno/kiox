from typing import Union

import numpy as np

from ..types import Action as ActionType
from ..types import Observation as ObservationType
from .proto.step_pb2 import (
    ContinuousAction,
    DiscreteAction,
    DType,
    Observation,
    Shape,
)


def convert_dtype_to_proto(dtype: np.dtype) -> DType:
    if dtype == np.uint8:
        return DType.UINT8
    if dtype == np.int32:
        return DType.INT32
    if dtype == np.float32:
        return DType.FLOAT32
    raise ValueError(f"invalid dtype: {dtype}")


def convert_proto_to_dtype(dtype: DType) -> np.dtype:
    if dtype == DType.UINT8:
        return np.uint8
    if dtype == DType.INT32:
        return np.int32
    if dtype == DType.FLOAT32:
        return np.float32
    raise ValueError(f"invalid dtype: {dtype}")


def convert_observation_to_proto(observation: ObservationType) -> Observation:
    if isinstance(observation, np.ndarray):
        observation = [observation]
    assert isinstance(observation, list)

    shapes = [Shape(dim=o.shape) for o in observation]
    data = [o.tobytes() for o in observation]
    dtypes = [convert_dtype_to_proto(o.dtype) for o in observation]

    return Observation(
        length=len(observation), shape=shapes, data=data, dtype=dtypes
    )


def convert_proto_to_observation(observation: Observation) -> ObservationType:
    observations = []
    for i in range(observation.length):
        dtype = convert_proto_to_dtype(observation.dtype[i])
        array = np.frombuffer(observation.data[i], dtype=dtype)
        observations.append(np.reshape(array, observation.shape[i].dim))
    if len(observations) == 1:
        return observations[0]
    return observations


def convert_action_to_proto(
    action: ActionType,
) -> Union[ContinuousAction, DiscreteAction]:
    if isinstance(action, np.ndarray):
        return ContinuousAction(length=action.size, data=action.tolist())
    if isinstance(action, int):
        return DiscreteAction(data=action)
    raise ValueError(f"invalid action type: {type(action)}")


def convert_proto_to_action(
    action: Union[ContinuousAction, DiscreteAction]
) -> ActionType:
    # TODO: a little hacky
    if str(type(action)) == str(type(ContinuousAction())):
        return np.array(action.data, dtype=np.float32)
    if str(type(action)) == str(type(DiscreteAction())):
        return int(action.data)
    raise ValueError(f"invalid action type: {type(action)}")
