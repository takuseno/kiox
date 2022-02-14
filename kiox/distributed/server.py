from concurrent import futures
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict

import grpc

from kiox.distributed.proto.step_pb2_grpc import (
    StepServiceServicer,
    add_StepServiceServicer_to_server,
)

from ..io import dump_memory, load_memory
from ..step_buffer import StepBuffer
from ..step_collector import StepCollector
from ..transition_buffer import TransitionBuffer
from ..transition_factory import TransitionFactory
from .proto.step_pb2 import StepProto, StepReply
from .utility import convert_proto_to_action, convert_proto_to_observation


class KioxStepServiceServicer(StepServiceServicer):  # type: ignore
    _step_buffer: StepBuffer
    _transition_buffer: TransitionBuffer
    _transition_factory: TransitionFactory
    _step_collectors: Dict[int, StepCollector]
    _n_steps: int
    _gamma: float

    def __init__(
        self,
        step_buffer: StepBuffer,
        transition_buffer: TransitionBuffer,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self._step_buffer = step_buffer
        self._transition_buffer = transition_buffer
        self._transition_factory = transition_factory
        self._step_collectors = {}
        self._n_steps = n_steps
        self._gamma = gamma

    def Send(self, request: StepProto, context: Any) -> StepReply:
        observation = convert_proto_to_observation(request.observation)
        if request.WhichOneof("action") == "discrete_action":
            action_proto = request.discrete_action
        else:
            action_proto = request.continuous_action
        action = convert_proto_to_action(action_proto)
        reward = request.reward
        terminal = request.terminal
        timeout = request.timeout
        rollout_id = request.rollout_id

        if rollout_id not in self._step_collectors:
            self._step_collectors[rollout_id] = StepCollector(
                step_buffer=self._step_buffer,
                transition_buffer=self._transition_buffer,
                transition_factory=self._transition_factory,
                n_steps=self._n_steps,
                gamma=self._gamma,
            )

        self._step_collectors[rollout_id].collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )

        if timeout:
            self._step_collectors[rollout_id].clip_episode()

        return StepReply(status="success")


ACK_START = "start"
ACK_ENDED = "ended"
ACK_SAVED = "saved"
ACK_LOADED = "loaded"
COMMAND_STOP = "stop"
COMMAND_SAMPLE = "sample"
COMMAND_GET_STEP_LEN = "get_step_len"
COMMAND_GET_TRANSITION_LEN = "get_transition_len"
COMMAND_SAVE = "save"
COMMAND_LOAD = "load"


def kiox_server_process(
    host: str,
    port: int,
    command_queue: "Queue[str]",
    ack_queue: "Queue[str]",
    step_buffer_builder: Callable[[], StepBuffer],
    transition_buffer_builder: Callable[[], TransitionBuffer],
    transition_factory_builder: Callable[[], TransitionFactory],
    max_workers: int = 10,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> None:
    step_buffer = step_buffer_builder()
    transition_buffer = transition_buffer_builder()
    transition_factory = transition_factory_builder()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = KioxStepServiceServicer(
        step_buffer=step_buffer,
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
            ack_queue.put(str(step_buffer.size()))
        elif command == COMMAND_GET_TRANSITION_LEN:
            ack_queue.put(str(transition_buffer.size()))
        elif command == COMMAND_SAVE:
            path = command_queue.get()
            with open(path, "wb") as f:
                dump_memory(f, step_buffer)
            ack_queue.put(ACK_SAVED)
        elif command == COMMAND_LOAD:
            path = command_queue.get()
            with open(path, "rb") as f:
                # create a temporary StepCollector
                step_collector = StepCollector(
                    step_buffer=step_buffer,
                    transition_buffer=transition_buffer,
                    transition_factory=transition_factory,
                    n_steps=n_steps,
                    gamma=gamma,
                )
                load_memory(f, step_collector)
            ack_queue.put(ACK_LOADED)
        else:
            raise ValueError(f"invalid command: {command}")

    server.stop(0)

    # return ack
    ack_queue.put(ACK_ENDED)


class KioxServer:
    _process: Process
    _command_queue: "Queue[str]"
    _ack_queue: "Queue[str]"

    def __init__(
        self,
        host: str,
        port: int,
        step_buffer_builder: Callable[[], StepBuffer],
        transition_buffer_builder: Callable[[], TransitionBuffer],
        transition_factory_builder: Callable[[], TransitionFactory],
        max_workers: int = 10,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> None:
        self._command_queue = Queue()
        self._ack_queue = Queue()
        self._process = Process(
            target=kiox_server_process,
            args=(
                host,
                port,
                self._command_queue,
                self._ack_queue,
                step_buffer_builder,
                transition_buffer_builder,
                transition_factory_builder,
                max_workers,
                n_steps,
                gamma,
            ),
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

    def save(self, path: str) -> None:
        self._command_queue.put(COMMAND_SAVE)
        self._command_queue.put(path)
        self._ack_queue.get()

    def load(self, path: str) -> None:
        self._command_queue.put(COMMAND_LOAD)
        self._command_queue.put(path)
        self._ack_queue.get()
