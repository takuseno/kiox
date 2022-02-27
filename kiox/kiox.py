from typing import BinaryIO, Union

from .batch_factory import Batch, BatchFactory
from .io import dump_memory, load_memory
from .item import Item
from .step_buffer import StepBuffer
from .step_collector import StepCollector
from .transition_buffer import TransitionBuffer
from .transition_factory import TransitionFactory


class Kiox:
    _transition_buffer: TransitionBuffer
    _step_buffer: StepBuffer
    _transition_factory: TransitionFactory
    _batch_factory: BatchFactory
    _step_collector: StepCollector

    def __init__(
        self,
        step_buffer: StepBuffer,
        transition_buffer: TransitionBuffer,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self._transition_buffer = transition_buffer
        self._step_buffer = step_buffer
        self._transition_factory = transition_factory
        self._batch_factory = BatchFactory(
            step_buffer=step_buffer,
            transition_buffer=transition_buffer,
        )
        self._step_collector = StepCollector(
            step_buffer=step_buffer,
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

    def clip_episode(self) -> None:
        self._step_collector.clip_episode()

    def sample(self, batch_size: int) -> Batch:
        return self._batch_factory.sample(batch_size)

    def copy_from(self, kiox: "Kiox") -> None:
        self._transition_buffer.copy_from(kiox.transition_buffer)
        self._step_buffer.copy_from(kiox.step_buffer)

    def save(self, f: BinaryIO) -> None:
        dump_memory(f, self._step_buffer)

    def load(self, f: BinaryIO) -> None:
        load_memory(f, self._step_collector)

    @property
    def transition_buffer(self) -> TransitionBuffer:
        return self._transition_buffer

    @property
    def step_buffer(self) -> StepBuffer:
        return self._step_buffer

    @property
    def transition_factory(self) -> TransitionFactory:
        return self._transition_factory
