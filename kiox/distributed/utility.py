import numpy as np

from ..item import Item
from .proto.step_pb2 import DType, ItemProto, Shape


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


def convert_item_to_proto(item: Item) -> ItemProto:
    if isinstance(item, (np.ndarray)):
        item = [item]
    elif isinstance(item, (int, float)):
        item = [np.array([item], dtype=np.float32)]
    assert isinstance(item, (list, tuple))

    shapes = [Shape(dim=el.shape) for el in item]
    data = [el.tobytes() for el in item]
    dtypes = [convert_dtype_to_proto(el.dtype) for el in item]

    return ItemProto(length=len(item), shape=shapes, data=data, dtype=dtypes)


def convert_proto_to_item(proto: ItemProto) -> Item:
    item = []
    for i in range(proto.length):
        dtype = convert_proto_to_dtype(proto.dtype[i])
        array = np.frombuffer(proto.data[i], dtype=dtype)
        item.append(np.reshape(array, proto.shape[i].dim))
    if len(item) == 1:
        return item[0]
    return item
