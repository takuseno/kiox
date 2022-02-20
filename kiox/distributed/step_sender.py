import dataclasses
from queue import Queue
from threading import Thread
from typing import Optional

import grpc

from ..types import Action as ActionType
from ..types import Observation as ObservationType
from .proto.step_pb2 import ContinuousAction, DiscreteAction, StepProto
from .proto.step_pb2_grpc import StepServiceStub
from .utility import convert_action_to_proto, convert_observation_to_proto


@dataclasses.dataclass(frozen=True)
class StepData:
    observation: ObservationType
    action: ActionType
    reward: float
    terminal: float
    timeout: Optional[bool]


class StepSender:
    _queue: "Queue[Optional[StepData]]"
    _thread: Thread

    def __init__(self, host: str, port: int, rollout_id: int):
        self._queue = Queue()
        self._thread = Thread(
            target=self._loop_thread,
            args=(host, port, rollout_id),
            daemon=True,
        )
        self._thread.start()

    def collect(
        self,
        observation: ObservationType,
        action: ActionType,
        reward: float,
        terminal: float,
        timeout: Optional[bool] = None,
    ) -> None:
        step_data = StepData(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            timeout=timeout,
        )
        self._queue.put(step_data)

    def _loop_thread(self, host: str, port: int, rollout_id: int) -> None:
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = StepServiceStub(channel)
        while True:
            step_data = self._queue.get()

            if step_data is None:
                break

            observation = convert_observation_to_proto(step_data.observation)
            action = convert_action_to_proto(step_data.action)
            timeout = (
                step_data.terminal
                if step_data.timeout is None
                else step_data.timeout
            )
            continuous_action = (
                action if isinstance(action, ContinuousAction) else None
            )
            discrete_action = (
                action if isinstance(action, DiscreteAction) else None
            )
            step = StepProto(
                observation=observation,
                continuous_action=continuous_action,
                discrete_action=discrete_action,
                reward=step_data.reward,
                terminal=step_data.terminal,
                timeout=timeout,
                rollout_id=rollout_id,
            )
            stub.Send(step)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()
