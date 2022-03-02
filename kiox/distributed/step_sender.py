import dataclasses
from queue import Queue
from threading import Thread
from typing import Optional, Union

import grpc

from ..item import Item
from .proto.step_pb2 import StepProto
from .proto.step_pb2_grpc import StepServiceStub
from .utility import convert_item_to_proto


@dataclasses.dataclass(frozen=True)
class StepData:
    """StepData data class.

    Args:
        observation: observation.
        action: action.
        reward: reward.
        terminal: terminal flag.
        timeout: timeout flag.

    """

    observation: Item
    action: Item
    reward: Item
    terminal: float
    timeout: Optional[bool]


class StepSender:
    """StepSender class.

    This class sends experience tuples via gRPC.

    .. code-block:: python

        sender = StepSender("localhost", 8000, 1)
        env = gym.make("CartPole-v0")

        obs = env.reset()
        while True:
            action = np.random.randint(2)
            next_obs, reward, terminal, _ = env.step(action)
            sender.collect(obs.astype(np.float32), action, reward, terminal)
            if terminal:
                break
            obs = next_obs
        sender.stop()

    Args:
        host: host address.
        port: port number.
        rollout_id: unique rollout worker id.

    """

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
        observation: Item,
        action: Item,
        reward: Item,
        terminal: Union[float, bool],
        timeout: Optional[bool] = None,
    ) -> None:
        """Sends experience tuple via gRPC.

        Args:
            observation: observation.
            action: action.
            reward: reward.
            terminal: terminal flag.
            timeout: timeout flag.

        """
        if not isinstance(terminal, float):
            terminal = float(terminal)
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

            observation = convert_item_to_proto(step_data.observation)
            action = convert_item_to_proto(step_data.action)
            reward = convert_item_to_proto(step_data.reward)
            timeout = (
                step_data.terminal
                if step_data.timeout is None
                else step_data.timeout
            )
            step = StepProto(
                observation=observation,
                action=action,
                reward=reward,
                terminal=step_data.terminal,
                timeout=timeout,
                rollout_id=rollout_id,
            )
            stub.Send(step)

    def stop(self) -> None:
        """Stops gRPC thread."""
        self._queue.put(None)
        self._thread.join()
