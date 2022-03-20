from typing import Dict, List, Optional, Sequence

import numpy as np

from .step import PartialStep, Step, StepBuffer
from .transition import LazyTransition
from .transition_buffer import TransitionBuffer


class Episode:
    """Episode class.

    This class represents a single episode with a sequence of steps.

    Args:
        step_buffer: StepBuffer object.
        transition_buffer: TransitionBuffer object.

    """

    _step_buffer: StepBuffer
    _transition_buffer: TransitionBuffer
    _transitions: List[LazyTransition]
    _idx_list: List[int]
    _prev_idx: Dict[int, int]
    _next_idx: Dict[int, int]
    _prev_step: Optional[Step]

    def __init__(
        self, step_buffer: StepBuffer, transition_buffer: TransitionBuffer
    ) -> None:
        self._step_buffer = step_buffer
        self._transition_buffer = transition_buffer
        self._transitions = []
        self._idx_list = []
        self._prev_idx = {}
        self._next_idx = {}
        self._prev_step = None

    def append_step(self, partial_step: PartialStep) -> Step:
        """Creates and stores Step object from PartialStep.

        Args:
            partial_step: PartialStep object.

        Returns:
            Step object.

        """
        step = self._step_buffer.append(partial_step)
        self._idx_list.append(step.idx)
        if self._prev_step:
            self._prev_idx[step.idx] = self._prev_step.idx
            self._next_idx[self._prev_step.idx] = step.idx
        self._prev_step = step
        return step

    def append_transition(
        self, transition: LazyTransition
    ) -> Optional[LazyTransition]:
        """Appends LazyTransition object.

        Args:
            transition: LazyTransition object.

        Returns:
            LazyTransition object dropped by TransitionBuffer.

        """
        self._transitions.append(transition)
        return self._transition_buffer.append(transition)

    def get(self, idx: int) -> Step:
        """Returns step by specified idx.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        return self._step_buffer.get(idx)

    def get_by_index(self, index: int) -> Step:
        """Returns step by specified index.

        Args:
            index: step index.

        Returns:
            Step object.

        """
        return self._step_buffer.get(self._idx_list[index])

    def get_next(self, idx: int, duration: int = 1) -> Optional[Step]:
        """Returns step ``duration`` steps ahead from ``idx``.

        If there is no more steps ahead, ``None`` will be returned.

        Args:
            idx: step idx.
            duration: the number of steps after ``idx``.

        Returns:
            Step object ``duration`` steps ahead.

        """
        next_idx = idx
        for _ in range(duration):
            if next_idx not in self._next_idx:
                return None
            next_idx = self._next_idx[next_idx]
        return self.get(next_idx)

    def get_prev(self, idx: int, duration: int = 1) -> Optional[Step]:
        """Returns step ``duration`` steps back from ``idx``.

        Args:
            idx: step idx.
            duration: the number of steps before ``idx``.

        Returns:
            Step object ``duration`` steps back.

        """
        prev_idx = idx
        for _ in range(duration):
            if prev_idx not in self._prev_idx:
                return None
            prev_idx = self._prev_idx[prev_idx]
        return self.get(prev_idx)

    def compute_return(
        self, idx: int, duration: int = 1, gamma: float = 0.99
    ) -> float:
        """Computes discounted return for ``duration`` steps.

        Args:
            idx: origin idx.
            duration: the number of steps after ``idx``.
            gamma: discounted factor.

        Returns:
            discounted return.

        """
        next_idx = idx
        ret = 0.0
        for i in range(duration):
            reward = self.get(next_idx).reward
            assert isinstance(reward, (float, np.ndarray))
            ret += (gamma**i) * reward
            if next_idx in self._next_idx:
                next_idx = self._next_idx[next_idx]
            else:
                break
        return ret

    def size(self) -> int:
        """Returns episode length.

        Returns:
            episode length.

        """
        return len(self._idx_list)

    def includes(self, idx: int) -> bool:
        """Returns if ``idx`` exists in episode.

        Returns:
            ``True`` if ``idx`` exists.

        """
        return idx in self._idx_list

    @property
    def steps(self) -> Sequence[Step]:
        return [self._step_buffer.get(idx) for idx in self._idx_list]

    @property
    def transitions(self) -> Sequence[LazyTransition]:
        return self._transitions


class EpisodeManager:
    """EpisodeManager class.

    This class takes a stream of steps and splits them into episodes.

    Args:
        step_buffer: StepBuffer object.
        transition_buffer: TransitionBuffer object.

    """

    _step_buffer: StepBuffer
    _transition_buffer: TransitionBuffer
    _episodes: List[Episode]
    _transition_to_episode: Dict[int, Episode]
    _dropped_transitions: Dict[Episode, List[LazyTransition]]

    def __init__(
        self, step_buffer: StepBuffer, transition_buffer: TransitionBuffer
    ) -> None:
        self._step_buffer = step_buffer
        self._transition_buffer = transition_buffer
        self._episodes = [Episode(step_buffer, transition_buffer)]
        self._transition_to_episode = {}
        self._dropped_transitions = {}

    def append_step(self, partial_step: PartialStep) -> Step:
        """Appends step to active episode.

        Args:
            partial_step: PartialStep object.

        Returns:
            Step object.

        """
        return self.active_episode.append_step(partial_step)

    def append_transition(self, transition: LazyTransition) -> None:
        """Appends LazyTransition object.

        If all transitions are dropped from an episode, the episode and
        included steps will be removed.

        Args:
            transition: LazyTransition object.

        """
        self._transition_to_episode[id(transition)] = self.active_episode
        dropped_transition = self.active_episode.append_transition(transition)
        if dropped_transition:
            episode = self._transition_to_episode[id(dropped_transition)]

            if episode not in self._dropped_transitions:
                self._dropped_transitions[episode] = []

            # record what transition has been removed
            self._dropped_transitions[episode].append(dropped_transition)
            del self._transition_to_episode[id(dropped_transition)]

            # remove steps and episode
            if len(self._dropped_transitions[episode]) == len(
                episode.transitions
            ):
                del self._dropped_transitions[episode]
                self._episodes.pop(self._episodes.index(episode))
                for step in episode.steps:
                    self._step_buffer.drop(step.idx)

    def get_step_by_idx(self, idx: int) -> Step:
        """Returns step by specified ``idx`.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        return self._step_buffer.get(idx)

    def clip_episode(self) -> None:
        """Clips active episode.

        This method should be called whenever episode reaches timeout or
        terminated.

        """
        self._episodes.append(
            Episode(self._step_buffer, self._transition_buffer)
        )

    def get_total_step_size(self) -> int:
        """Returns total step size.

        Returns:
            total step size.

        """
        return sum([e.size() for e in self._episodes])

    @property
    def active_episode(self) -> Episode:
        return self._episodes[-1]

    @property
    def episodes(self) -> Sequence[Episode]:
        return self._episodes
