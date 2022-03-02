from typing import Optional

import numpy as np

from .item import StackedItem, locate_stacked_item, sizeof_stacked_item
from .kiox import Kiox
from .transition_buffer import UnlimitedTransitionBuffer
from .transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
    TransitionFactory,
)


def build_from_dataset(
    observations: StackedItem,
    actions: StackedItem,
    rewards: np.ndarray,
    terminals: np.ndarray,
    transition_factory: TransitionFactory,
    timeouts: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> Kiox:
    transition_buffer = UnlimitedTransitionBuffer()
    kiox = Kiox(
        transition_factory=transition_factory,
        transition_buffer=transition_buffer,
        n_steps=n_steps,
        gamma=gamma,
    )
    for i in range(sizeof_stacked_item(observations)):
        kiox.collect(
            observation=locate_stacked_item(observations, i),
            action=locate_stacked_item(actions, i),
            reward=rewards[i],
            terminal=terminals[i],
        )
        if timeouts is not None and timeouts[i]:
            kiox.clip_episode()
    return kiox


def create_simple_kiox_from_dataset(
    observations: StackedItem,
    actions: StackedItem,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> Kiox:
    return build_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        transition_factory=SimpleTransitionFactory(),
        n_steps=n_steps,
        gamma=gamma,
    )


def create_frame_stack_kiox_from_dataset(
    observations: np.ndarray,
    actions: StackedItem,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
    n_frames: int = 1,
) -> Kiox:
    assert isinstance(
        observations, np.ndarray
    ), "supports image only observations"
    return build_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        transition_factory=FrameStackTransitionFactory(n_frames),
        n_steps=n_steps,
        gamma=gamma,
    )
