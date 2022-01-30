from typing import BinaryIO

import h5py

from .step_buffer import StepBuffer
from .step_collector import StepCollector


def dump_memory(f: BinaryIO, step_buffer: StepBuffer) -> None:
    # sort steps by its id
    steps = step_buffer.steps
    sorted_steps = sorted(steps, key=lambda s: s.idx)

    observations = []
    actions = []
    rewards = []
    terminals = []
    for step in sorted_steps:
        observations.append(step.observation)
        actions.append(step.action)
        rewards.append(step.reward)
        terminals.append(step.terminal)

    with h5py.File(f, "w") as h5:
        h5.create_dataset("observations", data=observations)
        h5.create_dataset("actions", data=actions)
        h5.create_dataset("rewards", data=rewards)
        h5.create_dataset("terminals", data=terminals)
        h5.flush()


def load_memory(f: BinaryIO, step_collector: StepCollector) -> None:
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]
        actions = h5["actions"][()]
        rewards = h5["rewards"][()]
        terminals = h5["terminals"][()]

    for observation, action, reward, terminal in zip(
        observations, actions, rewards, terminals
    ):
        step_collector.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )
