from typing import Dict, List, Optional, Sequence

import numpy as np

from .step import Step


class Episode:
    _buffer: List[Step]
    _index: Dict[int, int]
    _prev_index: Dict[int, int]
    _next_index: Dict[int, int]
    _prev_step: Optional[Step]
    _cursor: int

    def __init__(self) -> None:
        self._buffer = []
        self._index = {}
        self._prev_index = {}
        self._next_index = {}
        self._prev_step = None
        self._cursor = 0

    def append(self, step: Step) -> None:
        self._buffer.append(step)
        self._index[step.idx] = self._cursor
        if self._prev_step:
            self._prev_index[step.idx] = self._index[self._prev_step.idx]
            self._next_index[self._prev_step.idx] = self._cursor
        self._cursor += 1
        self._prev_step = step

    def get(self, idx: int) -> Step:
        assert idx in self._index, f"Step(idx={idx}) does not exist"
        return self._buffer[self._index[idx]]

    def get_by_index(self, index: int) -> Step:
        return self._buffer[index]

    def get_next(self, idx: int, duration: int = 1) -> Optional[Step]:
        next_idx = idx
        for _ in range(duration):
            if next_idx not in self._next_index:
                return None
            next_idx = self._buffer[self._next_index[next_idx]].idx
        return self.get(next_idx)

    def get_prev(self, idx: int, duration: int = 1) -> Optional[Step]:
        prev_idx = idx
        for _ in range(duration):
            if prev_idx not in self._prev_index:
                return None
            prev_idx = self._buffer[self._prev_index[prev_idx]].idx
        return self.get(prev_idx)

    def compute_return(
        self, idx: int, duration: int = 1, gamma: float = 0.99
    ) -> float:
        next_idx = idx
        ret = 0.0
        for i in range(duration):
            reward = self.get(next_idx).reward
            assert isinstance(reward, (float, np.ndarray))
            ret += (gamma**i) * reward
            if next_idx in self._next_index:
                next_idx = self._next_index[next_idx]
            else:
                break
        return ret

    def size(self) -> int:
        return len(self._buffer)

    def includes(self, idx: int) -> bool:
        return idx in self._index

    @property
    def steps(self) -> Sequence[Step]:
        return self._buffer


class EpisodeManager:
    _episodes: List[Episode]
    _step_to_episode: Dict[int, Episode]
    _episode_to_step: Dict[Episode, List[int]]
    _dropped_step: Dict[Episode, List[int]]

    def __init__(self) -> None:
        self._episodes = [Episode()]
        self._step_to_episode = {}
        self._episode_to_step = {}
        self._dropped_step = {}

    def append(self, step: Step) -> None:
        self.active_episode.append(step)
        self._step_to_episode[step.idx] = self.active_episode
        if self.active_episode not in self._episode_to_step:
            self._episode_to_step[self.active_episode] = []
        self._episode_to_step[self.active_episode].append(step.idx)

    def get_step_by_idx(self, idx: int) -> Step:
        return self._step_to_episode[idx].get(idx)

    def get_episode_by_step_idx(self, idx: int) -> Episode:
        return self._step_to_episode[idx]

    def clip_episode(self) -> None:
        self._episodes.append(Episode())

    def drop_step(self, idx: int) -> None:
        # get corresponding episode
        episode = self.get_episode_by_step_idx(idx)

        if episode not in self._dropped_step:
            self._dropped_step[episode] = []
        self._dropped_step[episode].append(idx)

        # drop episode if necessary
        if len(self._dropped_step[episode]) == episode.size():
            assert episode is not self.active_episode
            del self._dropped_step[episode]
            self.drop(episode)

    def drop(self, episode: Episode) -> None:
        for idx in self._episode_to_step[episode]:
            del self._step_to_episode[idx]
        del self._episode_to_step[episode]
        self._episodes.pop(self._episodes.index(episode))

    def copy_from(self, episode_manager: "EpisodeManager") -> None:
        for episode in episode_manager.episodes:
            for step in episode.steps:
                self.append(step)
            self.clip_episode()

    def get_total_step_size(self) -> int:
        return sum([e.size() for e in self._episodes])

    @property
    def active_episode(self) -> Episode:
        return self._episodes[-1]

    @property
    def episodes(self) -> Sequence[Episode]:
        return self._episodes
