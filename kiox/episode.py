from typing import Dict, List, Optional, Sequence

import numpy as np

from .step import Step, StepBuffer


class Episode:
    """Episode class.

    This class represents a single episode with a sequence of steps.

    Args:
        step_buffer: StepBuffer object.

    """

    _buffer: StepBuffer
    _idx_list: List[int]
    _prev_idx: Dict[int, int]
    _next_idx: Dict[int, int]
    _prev_step: Optional[Step]

    def __init__(self, step_buffer: StepBuffer) -> None:
        self._buffer = step_buffer
        self._idx_list = []
        self._prev_idx = {}
        self._next_idx = {}
        self._prev_step = None

    def append(self, step: Step) -> None:
        """Appends step.

        Args:
            step: Step object.

        """
        self._buffer.append(step)
        self._idx_list.append(step.idx)
        if self._prev_step:
            self._prev_idx[step.idx] = self._prev_step.idx
            self._next_idx[self._prev_step.idx] = step.idx
        self._prev_step = step

    def get(self, idx: int) -> Step:
        """Returns step by specified idx.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        return self._buffer.get(idx)

    def get_by_index(self, index: int) -> Step:
        """Returns step by specified index.

        Args:
            index: step index.

        Returns:
            Step object.

        """
        return self._buffer.get(self._idx_list[index])

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
        return [self._buffer.get(idx) for idx in self._idx_list]


class EpisodeManager:
    """EpisodeManager class.

    This class takes a stream of steps and splits them into episodes.

    Args:
        step_buffer: StepBuffer object.

    """

    _step_buffer: StepBuffer
    _episodes: List[Episode]
    _step_to_episode: Dict[int, Episode]
    _dropped_step: Dict[Episode, List[int]]

    def __init__(self, step_buffer: StepBuffer) -> None:
        self._step_buffer = step_buffer
        self._episodes = [Episode(step_buffer)]
        self._step_to_episode = {}
        self._dropped_step = {}

    def append(self, step: Step) -> None:
        """Appends step to active episode.

        Args:
            step: Step object.

        """
        self.active_episode.append(step)
        self._step_to_episode[step.idx] = self.active_episode

    def get_step_by_idx(self, idx: int) -> Step:
        """Returns step by specified ``idx`.

        Args:
            idx: step idx.

        Returns:
            Step object.

        """
        return self._step_to_episode[idx].get(idx)

    def get_episode_by_step_idx(self, idx: int) -> Episode:
        """Returns episode including step of ``idx``.

        Args:
            idx: step idx.

        Returns:
            Episode object.

        """
        return self._step_to_episode[idx]

    def clip_episode(self) -> None:
        """Clips active episode.

        This method should be called whenever episode reaches timeout or
        terminated.

        """
        self._episodes.append(Episode(self._step_buffer))

    def drop_step(self, idx: int) -> None:
        """Virtually drops step by specified ``idx``.

        This method does not actually drop the step until all steps are dropped
        from an episode.

        Args:
            idx: step idx.

        """
        # get corresponding episode
        episode = self.get_episode_by_step_idx(idx)

        if episode not in self._dropped_step:
            self._dropped_step[episode] = []
        self._dropped_step[episode].append(idx)

        # drop episode if necessary
        if len(self._dropped_step[episode]) == episode.size():
            assert episode is not self.active_episode

            # drop from StepBuffer and step-to-episode mapping
            for step in episode.steps:
                del self._step_to_episode[step.idx]
                self._step_buffer.drop(step.idx)

            # drop from dropped mapping
            del self._dropped_step[episode]

            # drop episode
            self._episodes.pop(self._episodes.index(episode))

    def copy_from(self, episode_manager: "EpisodeManager") -> None:
        """Copes steps from another EpisodeManager.

        Args:
            episode_manager: source episodes.

        """
        for episode in episode_manager.episodes:
            for step in episode.steps:
                self.append(step)
            self.clip_episode()

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
