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
    """Kiox class.

    This class takes a single stream of experiences.

    .. code-block:: python

        kiox = Kiox(FIFOTransitionBuffer(1000), SimpleTransitionFactory())

        # collect data
        env = gym.make("CartPole-v0")
        obs = env.reset()
        while True:
            action = np.random.randint(2)
            next_obs, reward, terminal, _ = env.step(action)
            kiox.collect(obs, action, reward, terminal)
            if terminal:
                break
            obs = next_obs

        # sample mini-batch
        batch = kiox.sample(32)

        # save data
        with open("data.h5", "wb") as f:
            kiox.save(f)

        # load data
        kiox2 = Kiox(FIFOTransitionBuffer(1000), SimpleTransitionFactory())
        with open("data.h5", "rb") as f:
            kiox2.load(f)

    Args:
        transition_buffer: TransitionBuffer object.
        transition_factory: TransitionFactory object.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    """

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
        """Stores experience tuple.

        Args:
            observation: observation.
            action: action.
            reward: reward.
            terminal: terminal flag.

        """
        terminal = float(terminal) if isinstance(terminal, bool) else terminal
        self._step_collector.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )

    def get_step_buffer_size(self) -> int:
        """Returns number of stored tuples.

        Returns:
            number of tuples.

        """
        return self._step_buffer.size()

    def get_transition_buffer_size(self) -> int:
        """Returns number of stored transitions.

        Returns:
            number of transitions.

        """
        return self._transition_buffer.size()

    def clip_episode(self) -> None:
        """Clips active episode.

        This method should be called whenever the current episode ends for
        timeout.

        """
        self._step_collector.clip_episode()

    def sample(self, batch_size: int) -> Batch:
        """Samples transitions and returns mini-batch.

        Args:
            batch_size: batch size.

        Returns:
            mini-batch.

        """
        return self._batch_factory.sample(batch_size)

    def copy_from(self, kiox: "Kiox") -> None:
        """Copies steps and transitions from another Kiox object.

        Args:
            kiox: source Kiox object.

        """
        self._transition_buffer.copy_from(kiox.transition_buffer)
        self._episode_manager.copy_from(kiox.episode_manager)

    def save(self, f: BinaryIO) -> None:
        """Saves data as HDF5.

        Args:
            f: I/O-like object.

        """
        dump_memory(f, self._episode_manager.episodes)

    def load(self, f: BinaryIO) -> None:
        """Loads HDF5 data.

        Args:
            f: I/O-like object.

        """
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
