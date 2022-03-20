from typing import Optional

from .episode import EpisodeManager
from .item import Item
from .step import PartialStep
from .transition_factory import TransitionFactory


class StepCollector:
    """StepCollector class.

    This class takes a single stream of experience tuples.
    The given tuples are converted to Step object and LazyTransition object.

    Args:
        episode_manager: EpisodeManager object.
        transition_factory: TransitionFactory object.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    """

    _transition_factory: TransitionFactory
    _episode_manager: EpisodeManager
    _n_steps: int
    _gamma: float

    def __init__(
        self,
        episode_manager: EpisodeManager,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self._episode_manager = episode_manager
        self._transition_factory = transition_factory
        self._n_steps = n_steps
        self._gamma = gamma

    def collect(
        self,
        observation: Item,
        action: Item,
        reward: Item,
        terminal: float,
        timeout: Optional[bool] = None,
    ) -> None:
        """Stores experience tuples and Step and Transition.

        Args:
            observation: observation.
            action: action.
            reward: reward.
            terminal: terminal flag.
            timeout: timeout flag.

        """
        partial_step = PartialStep(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )
        step = self._episode_manager.append_step(partial_step)

        if self._episode_manager.active_episode.size() > self._n_steps:
            last_step = self._episode_manager.active_episode.get_prev(
                step.idx, self._n_steps
            )
            assert last_step
            transition = self._transition_factory.create(
                step=last_step,
                next_step=step,
                episode=self._episode_manager.active_episode,
                duration=self._n_steps,
                gamma=self._gamma,
            )
            self._episode_manager.append_transition(transition)

        if terminal:
            # consume remaining steps
            for i in reversed(range(self._n_steps)):
                last_step = self._episode_manager.active_episode.get_prev(
                    step.idx, i
                )
                assert last_step
                transition = self._transition_factory.create(
                    step=last_step,
                    next_step=None,
                    episode=self._episode_manager.active_episode,
                    duration=i + 1,
                    gamma=self._gamma,
                )
                self._episode_manager.append_transition(transition)

        if terminal or timeout:
            self.clip_episode()

    def clip_episode(self) -> None:
        """Clips active episode.

        This method should be called whenever the current episode reaches
        timeout or terminated.

        """
        self._episode_manager.clip_episode()

    @property
    def episode_manager(self) -> EpisodeManager:
        return self._episode_manager
