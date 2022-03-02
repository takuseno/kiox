from typing import BinaryIO, Sequence

import h5py

from .episode import Episode
from .step_collector import StepCollector


def dump_memory(f: BinaryIO, episodes: Sequence[Episode]) -> None:
    """Dumps data as HDF5.

    Args:
        f: I/O-like object.
        episodes: list of episodes.

    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    for episode in episodes:
        for i, step in enumerate(episode.steps):
            observations.append(step.observation)
            actions.append(step.action)
            rewards.append(step.reward)
            terminals.append(step.terminal)
            if step.terminal:
                timeouts.append(False)
            elif i == episode.size() - 1:
                timeouts.append(True)
            else:
                timeouts.append(False)

    with h5py.File(f, "w") as h5:
        h5.create_dataset("observations", data=observations)
        h5.create_dataset("actions", data=actions)
        h5.create_dataset("rewards", data=rewards)
        h5.create_dataset("terminals", data=terminals)
        h5.create_dataset("timeouts", data=terminals)
        h5.flush()


def load_memory(f: BinaryIO, step_collector: StepCollector) -> None:
    """Loads HDF5 data.

    Args:
        f: I/O-like object.
        step_collector: StepCollector object.

    """
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]
        actions = h5["actions"][()]
        rewards = h5["rewards"][()]
        terminals = h5["terminals"][()]
        timeouts = h5["timeouts"][()]

    for observation, action, reward, terminal, timeout in zip(
        observations, actions, rewards, terminals, timeouts
    ):
        step_collector.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            timeout=timeout,
        )
