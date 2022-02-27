from concurrent import futures
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, Sequence, Union

import grpc

from kiox.distributed.proto.step_pb2_grpc import (
    StepServiceServicer,
    add_StepServiceServicer_to_server,
)

from ..batch_factory import Batch
from ..episode import EpisodeManager
from ..io import dump_memory, load_memory
from ..step_collector import StepCollector
from ..transition_buffer import TransitionBuffer
from ..transition_factory import TransitionFactory
from .proto.step_pb2 import StepProto, StepReply
from .shared_batch_factory import SharedBatchFactory
from .utility import convert_proto_to_item


class KioxStepServiceServicer(StepServiceServicer):  # type: ignore
    _episode_manager: EpisodeManager
    _transition_buffer: TransitionBuffer
    _transition_factory: TransitionFactory
    _step_collectors: Dict[int, StepCollector]
    _n_steps: int
    _gamma: float

    def __init__(
        self,
        episode_manager: EpisodeManager,
        transition_buffer: TransitionBuffer,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self._episode_manager = episode_manager
        self._transition_buffer = transition_buffer
        self._transition_factory = transition_factory
        self._step_collectors = {}
        self._n_steps = n_steps
        self._gamma = gamma

    def Send(self, request: StepProto, context: Any) -> StepReply:
        observation = convert_proto_to_item(request.observation)
        action = convert_proto_to_item(request.action)
        reward = convert_proto_to_item(request.reward)
        timeout = request.timeout
        rollout_id = request.rollout_id

        if rollout_id not in self._step_collectors:
            self._step_collectors[rollout_id] = StepCollector(
                episode_manager=self._episode_manager,
                transition_buffer=self._transition_buffer,
                transition_factory=self._transition_factory,
                n_steps=self._n_steps,
                gamma=self._gamma,
            )

        self._step_collectors[rollout_id].collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=request.terminal,
        )

        if timeout:
            self._step_collectors[rollout_id].clip_episode()

        return StepReply(status="success")


ACK_START = "start"
ACK_ENDED = "ended"
ACK_SAVED = "saved"
ACK_LOADED = "loaded"
ACK_SAMPLED = "sampled"
COMMAND_STOP = "stop"
COMMAND_SAMPLE = "sample"
COMMAND_GET_STEP_LEN = "get_step_len"
COMMAND_GET_TRANSITION_LEN = "get_transition_len"
COMMAND_SAVE = "save"
COMMAND_LOAD = "load"


def kiox_server_process(
    host: str,
    port: int,
    batch_factory: SharedBatchFactory,
    command_queue: "Queue[str]",
    ack_queue: "Queue[str]",
    transition_buffer_builder: Callable[[], TransitionBuffer],
    transition_factory_builder: Callable[[], TransitionFactory],
    max_workers: int = 10,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> None:
    episode_manager = EpisodeManager()
    transition_buffer = transition_buffer_builder()
    transition_factory = transition_factory_builder()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = KioxStepServiceServicer(
        episode_manager=episode_manager,
        transition_buffer=transition_buffer,
        transition_factory=transition_factory,
        n_steps=n_steps,
        gamma=gamma,
    )
    add_StepServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    # return ack
    ack_queue.put(ACK_START)

    while True:
        command = command_queue.get()
        if command == COMMAND_STOP:
            break
        if command == COMMAND_GET_STEP_LEN:
            ack_queue.put(str(episode_manager.get_total_step_size()))
        elif command == COMMAND_GET_TRANSITION_LEN:
            ack_queue.put(str(transition_buffer.size()))
        elif command == COMMAND_SAVE:
            path = command_queue.get()
            with open(path, "wb") as f:
                dump_memory(f, episode_manager)
            ack_queue.put(ACK_SAVED)
        elif command == COMMAND_LOAD:
            path = command_queue.get()
            with open(path, "rb") as f:
                # create a temporary StepCollector
                step_collector = StepCollector(
                    episode_manager=episode_manager,
                    transition_buffer=transition_buffer,
                    transition_factory=transition_factory,
                    n_steps=n_steps,
                    gamma=gamma,
                )
                load_memory(f, step_collector)
            ack_queue.put(ACK_LOADED)
        elif command == COMMAND_SAMPLE:
            batch_factory.sample(episode_manager, transition_buffer)
            ack_queue.put(ACK_SAMPLED)
        else:
            raise ValueError(f"invalid command: {command}")

    server.stop(0)

    # return ack
    ack_queue.put(ACK_ENDED)


class KioxServer:
    _batch_factory: SharedBatchFactory
    _process: Process
    _command_queue: "Queue[str]"
    _ack_queue: "Queue[str]"

    def __init__(
        self,
        host: str,
        port: int,
        observation_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        action_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        reward_shape: Union[Sequence[Sequence[int]], Sequence[int]],
        batch_size: int,
        transition_buffer_builder: Callable[[], TransitionBuffer],
        transition_factory_builder: Callable[[], TransitionFactory],
        max_workers: int = 10,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> None:
        self._batch_factory = SharedBatchFactory(
            observation_shape=observation_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            batch_size=batch_size,
        )
        self._command_queue = Queue()
        self._ack_queue = Queue()
        self._process = Process(
            target=kiox_server_process,
            args=(
                host,
                port,
                self._batch_factory,
                self._command_queue,
                self._ack_queue,
                transition_buffer_builder,
                transition_factory_builder,
                max_workers,
                n_steps,
                gamma,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()
        self._ack_queue.get()

    def stop(self) -> None:
        self._command_queue.put(COMMAND_STOP)
        self._ack_queue.get()

    def get_step_buffer_size(self) -> int:
        self._command_queue.put(COMMAND_GET_STEP_LEN)
        return int(self._ack_queue.get())

    def get_transition_buffer_size(self) -> int:
        self._command_queue.put(COMMAND_GET_TRANSITION_LEN)
        return int(self._ack_queue.get())

    def sample(self) -> Batch:
        self._command_queue.put(COMMAND_SAMPLE)
        self._ack_queue.get()
        return self._batch_factory.batch

    def save(self, path: str) -> None:
        self._command_queue.put(COMMAND_SAVE)
        self._command_queue.put(path)
        self._ack_queue.get()

    def load(self, path: str) -> None:
        self._command_queue.put(COMMAND_LOAD)
        self._command_queue.put(path)
        self._ack_queue.get()
