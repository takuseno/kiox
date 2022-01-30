from typing import Optional

import numpy as np

from .kiox import Kiox
from .step_buffer import UnlimitedStepBuffer
from .step_collector import StepCollector
from .transition_buffer import UnlimitedTransitionBuffer
from .transition_factory import (
    FrameStackTransitionFactory,
    SimpleTransitionFactory,
    TransitionFactory,
)


def build_from_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    transition_factory: TransitionFactory,
    episode_terminals: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> Kiox:
    step_buffer = UnlimitedStepBuffer()
    transition_buffer = UnlimitedTransitionBuffer()
    step_collector = StepCollector(
        step_buffer=step_buffer,
        transition_buffer=transition_buffer,
        transition_factory=transition_factory,
        n_steps=n_steps,
        gamma=gamma,
    )
    for i in range(observations.shape[0]):
        step_collector.collect(
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            terminal=terminals[i],
        )
        if episode_terminals is not None and episode_terminals[i]:
            step_collector.clip_episode()
    return Kiox(
        step_buffer=step_buffer,
        transition_factory=transition_factory,
        transition_buffer=transition_buffer,
        n_steps=n_steps,
    )


def create_simple_kiox_from_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    episode_terminals: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
) -> Kiox:
    return build_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=episode_terminals,
        transition_factory=SimpleTransitionFactory(),
        n_steps=n_steps,
        gamma=gamma,
    )


def create_frame_stack_kiox_from_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    episode_terminals: Optional[np.ndarray] = None,
    n_steps: int = 1,
    gamma: float = 0.99,
    n_frames: int = 1,
) -> Kiox:
    return build_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=episode_terminals,
        transition_factory=FrameStackTransitionFactory(n_frames),
        n_steps=n_steps,
        gamma=gamma,
    )
