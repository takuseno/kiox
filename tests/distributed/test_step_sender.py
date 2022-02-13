import time
from concurrent import futures

import grpc
import numpy as np

from kiox.distributed.proto.step_pb2 import StepReply
from kiox.distributed.proto.step_pb2_grpc import (
    StepServiceServicer,
    add_StepServiceServicer_to_server,
)
from kiox.distributed.step_sender import StepSender


class DummyServiceServicer(StepServiceServicer):
    def __init__(self, obs_shape, action_shape):
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def Send(self, request, context):
        assert request.observation.shape[0].dim == self.obs_shape
        assert request.continuous_action.length == self.action_shape[0]
        return StepReply(status="success")


def test_step_sender():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = DummyServiceServicer([3, 84, 84], [4])
    add_StepServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:8000")
    server.start()

    sender = StepSender("localhost", 8000, 1)
    observation = np.random.random((3, 84, 84)).astype(np.float32)
    action = np.random.random(4)
    reward = np.random.random()

    sender.collect(
        observation=observation,
        action=action,
        reward=reward,
        terminal=0.0,
    )

    time.sleep(1.0)

    server.stop(0)
    sender.stop()
