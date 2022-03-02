from concurrent import futures
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Sequence, Union

import grpc

from kiox.distributed.proto.step_pb2_grpc import (
    StepServiceServicer,
    add_StepServiceServicer_to_server,
)

from ..batch_factory import Batch
from ..episode import Episode, EpisodeManager
from ..io import dump_memory, load_memory
from ..step import StepBuffer
from ..step_collector import StepCollector
from ..transition_buffer import TransitionBuffer
from ..transition_factory import TransitionFactory
from .proto.step_pb2 import StepProto, StepReply
from .shared_batch_factory import SharedBatchFactory
from .utility import convert_proto_to_item

IDX_OFFSET = 100000000


class KioxStepServiceServicer(StepServiceServicer):  # type: ignore
    """KioxStepServiceServicer class.

    This class is a gRPC endpoint to receive remote steps.

    Args:
        step_buffer: StepBuffer object.
        transition_buffer: TransitionBuffer object.
        transition_factory: TransitionFactory object.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    """

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
        """gRPC endpooint for Send.

        This endpoint receives experience tuples with ``rollout_id`` parameter.
        If there are no tuples from ``rollout_id``, new ``StepCollector`` will
        be created.

        Args:
            request: protocol buffer step.
            context: context info.

        Returns:
            protocol buffer reply.

        """
        observation = convert_proto_to_item(request.observation)
        action = convert_proto_to_item(request.action)
        reward = convert_proto_to_item(request.reward)
        timeout = request.timeout
        rollout_id = request.rollout_id

        if rollout_id < 0:
            return StepReply(status="rollout_id must be positive integer.")

        if rollout_id not in self._step_collectors:
            self.append_step_collector(rollout_id, self._step_buffer)

        self._step_collectors[rollout_id].collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=request.terminal,
        )

        if timeout:
            self._step_collectors[rollout_id].clip_episode()

        return StepReply(status="success")

    def append_step_collector(
        self, rollout_id: int, step_buffer: StepBuffer
    ) -> None:
        """Creates StepCollector object for ``rollout_id``.

        Args:
            rollout_id: rollout worker id.
            step_buffer: StepBuffer object.

        """
        assert rollout_id not in self._step_collectors
        self._step_collectors[rollout_id] = StepCollector(
            episode_manager=EpisodeManager(step_buffer),
            transition_buffer=self._transition_buffer,
            transition_factory=self._transition_factory,
            n_steps=self._n_steps,
            gamma=self._gamma,
            idx_offset=len(self._step_collectors) * IDX_OFFSET,
        )

    def get_step_collector_by_rollout_id(
        self, rollout_id: int
    ) -> StepCollector:
        """Returns StepCollector by specified ``rollout_id``.

        Args:
            rollout_id: rollout worker id.

        Returns:
            StepCollector object.

        """
        return self._step_collectors[rollout_id]

    def has_step_collector(self, rollout_id: int) -> bool:
        """Returns if StepCollector object exists for ``rollout_id``.

        Args:
            rollout_id: rollout worker id.

        Returns:
            ``True`` if StepCollector exists for ``rollout_id``.

        """
        return rollout_id in self._step_collectors

    @property
    def episode_managers(self) -> Sequence[EpisodeManager]:
        return [sc.episode_manager for sc in self._step_collectors.values()]


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
    """Child process for server loop.

    Args:
        host: host address.
        port: port number.
        batch_factory: SharedBatchFactory object.
        command_queue: queue from parent process.
        ack_queue: queue from child process.
        transition_buffer_builder: function to build TransitionBuffer object.
        transition_factory_builder: function to build TransitionFactory object.
        max_workers: maximum number of threads for gRPC.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    """
    step_buffer = StepBuffer()
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
            episodes: List[Episode] = []
            for episode_manager in servicer.episode_managers:
                episodes.extend(episode_manager.episodes)
            with open(path, "wb") as f:
                dump_memory(f, episodes)
            ack_queue.put(ACK_SAVED)
        elif command == COMMAND_LOAD:
            path = command_queue.get()
            with open(path, "rb") as f:
                # create a special StepCollector
                if not servicer.has_step_collector(-1):
                    servicer.append_step_collector(-1, step_buffer)
                load_memory(f, servicer.get_step_collector_by_rollout_id(-1))
            ack_queue.put(ACK_LOADED)
        elif command == COMMAND_SAMPLE:
            batch_factory.sample(step_buffer, transition_buffer)
            ack_queue.put(ACK_SAMPLED)
        else:
            raise ValueError(f"invalid command: {command}")

    server.stop(0)

    # return ack
    ack_queue.put(ACK_ENDED)


class KioxServer:
    """KioxServer class.

    This class launches gRPC server in a child process to receive remote steps
    and samples mini-batch.

    .. code-block:: python

        def rollout():
            sender = StepSender("localhost", 8000, 1)
            env = gym.make("CartPole-v0")

            for _ in range(1000):
                obs = env.reset()
                while True:
                    action = np.random.randint(2)
                    next_obs, reward, terminal, _ = env.step(action)
                    sender.collect(
                        observation=obs.astype(np.float32),
                        action=action,
                        reward=reward,
                        terminal=terminal,
                    )
                    if terminal:
                        break
                    obs = next_obs
            sender.stop()

        server = KioxServer(
            host="localhost",
            port=8000,
            observation_shape=(4,),
            action_shape=(1,),
            reward_shape=(1,),
            batch_size=32,
            transition_buffer_builder=lambda: FIFOTransitionBuffer(1000),
            transition_factory_builder=lambda: SimpleTransitionFactory(),
        )
        server.start()

        # start rollout
        p = Process(target=rollout)
        p.start()

        # sample mini-batch
        batch = server.sample()
        assert batch.observations.shape == (32, 4)

        # save data
        server.save("data.h5")

        # load data
        server2 = KioxServer(
            host="localhost",
            port=9000,
            observation_shape=(4,),
            action_shape=(1,),
            reward_shape=(1,),
            batch_size=32,
            transition_buffer_builder=lambda: FIFOTransitionBuffer(1000),
            transition_factory_builder=lambda: SimpleTransitionFactory(),
        )
        server2.start()
        server2.load("data.h5")

    Args:
        host: host address.
        port: port number.
        observation_shape: shape of observation.
        action_shape: shape of action.
        reward_shape: shape of reward.
        batch_size: batch size.
        transition_buffer_builder: function to build TransitionBuffer object.
        transition_factory_builder: function to build TransitionFactory object.
        max_workers: maximum number of workers for gRPC.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    """

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
        """Starts gRPC server process."""
        self._process.start()
        self._ack_queue.get()

    def stop(self) -> None:
        """Stops gRPC server process."""
        self._command_queue.put(COMMAND_STOP)
        self._ack_queue.get()

    def get_step_buffer_size(self) -> int:
        """Returns number of stored steps.

        Returns:
            number of stored steps.

        """
        self._command_queue.put(COMMAND_GET_STEP_LEN)
        return int(self._ack_queue.get())

    def get_transition_buffer_size(self) -> int:
        """Returns number of stored transitions.

        Returns:
            number of stored transitions.

        """
        self._command_queue.put(COMMAND_GET_TRANSITION_LEN)
        return int(self._ack_queue.get())

    def sample(self) -> Batch:
        """Samples transitions and returns mini-batch.

        Returns:
            mini-batch.

        """
        self._command_queue.put(COMMAND_SAMPLE)
        self._ack_queue.get()
        return self._batch_factory.batch

    def save(self, path: str) -> None:
        """Saves data as HDF5 file to disk.

        Args:
            path: path to save.

        """
        self._command_queue.put(COMMAND_SAVE)
        self._command_queue.put(path)
        self._ack_queue.get()

    def load(self, path: str) -> None:
        """Loads HDF5 data from disk.

        Args:
            path: path to load.

        """
        self._command_queue.put(COMMAND_LOAD)
        self._command_queue.put(path)
        self._ack_queue.get()
