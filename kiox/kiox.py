from typing import BinaryIO, Union

from .batch_factory import Batch, BatchFactory
from .episode import EpisodeManager
from .io import dump_memory, load_memory
from .item import Item
from .step import StepBuffer
from .step_collector import StepCollector
from .transition_buffer import TransitionBuffer
from .transition_factory import TransitionFactory


class Kiox:
    _step_buffer: StepBuffer
    _episode_manager: EpisodeManager
    _transition_buffer: TransitionBuffer
    _transition_factory: TransitionFactory
    _batch_factory: BatchFactory
    _step_collector: StepCollector

    def __init__(
        self,
        transition_buffer: TransitionBuffer,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self._step_buffer = StepBuffer()
        self._episode_manager = EpisodeManager(self._step_buffer)
        self._transition_buffer = transition_buffer
        self._transition_factory = transition_factory
        self._batch_factory = BatchFactory(
            step_buffer=self._step_buffer,
            transition_buffer=transition_buffer,
        )
        self._step_collector = StepCollector(
            episode_manager=self._episode_manager,
            transition_buffer=transition_buffer,
            transition_factory=transition_factory,
            n_steps=n_steps,
            gamma=gamma,
        )

    def collect(
        self,
        observation: Item,
        action: Item,
        reward: Item,
        terminal: Union[bool, float],
    ) -> None:
        terminal = float(terminal) if isinstance(terminal, bool) else terminal
        self._step_collector.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )

    def get_step_buffer_size(self) -> int:
        return self._step_buffer.size()

    def get_transition_buffer_size(self) -> int:
        return self._transition_buffer.size()

    def clip_episode(self) -> None:
        self._step_collector.clip_episode()

    def sample(self, batch_size: int) -> Batch:
        return self._batch_factory.sample(batch_size)

    def copy_from(self, kiox: "Kiox") -> None:
        self._transition_buffer.copy_from(kiox.transition_buffer)
        self._episode_manager.copy_from(kiox.episode_manager)

    def save(self, f: BinaryIO) -> None:
        dump_memory(f, self._episode_manager.episodes)

    def load(self, f: BinaryIO) -> None:
        load_memory(f, self._step_collector)

    @property
    def episode_manager(self) -> EpisodeManager:
        return self._episode_manager

    @property
    def transition_buffer(self) -> TransitionBuffer:
        return self._transition_buffer

    @property
    def transition_factory(self) -> TransitionFactory:
        return self._transition_factory
