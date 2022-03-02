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
    """Builds Kiox object from pre-collected data.

    .. code-block:: python

        # dataset
        observations = np.random.random((1000, 100))
        actions = np.random.random((1000, 4))
        rewards = np.random.random(1000)
        terminals = np.zeros(1000)

        # build Kiox
        kiox = build_from_dataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            transition_factory=SimpleTransitionFactory(),
        )

        # sample mini-batch
        batch = kiox.sample(32)
        assert batch.observations.shape == (32, 100)

    Args:
        observations: a sequence of observations.
        actions: a sequence of actions.
        rewards: a sequence of rewards.
        terminals: a sequence of terminals.
        transition_factory: TransitionFactory object.
        timeouts: a sequence of timeout flags.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    Returns:
        Kiox object.

    """
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
    """Builds Kiox with SimpleTransitionFactory.

    This method is an alias of ``build_from_dataset`` with
    ``SimpleTransitionFactory``.

    Args:
        observations: a sequence of observations.
        actions: a sequence of actions.
        rewards: a sequence of rewards.
        terminals: a sequence of terminals.
        timeouts: a sequence of timeout flags.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    Returns:
        Kiox object.

    """
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
    """Builds Kiox with FrameStackTransitionFactory.

    This method is an alias of ``build_from_dataset`` with
    ``FrameStackTransitionFactory``.

    Args:
        observations: a sequence of observations.
        actions: a sequence of actions.
        rewards: a sequence of rewards.
        terminals: a sequence of terminals.
        timeouts: a sequence of timeout flags.
        n_steps: step size for multi-step learning. This corresponds to TD(N).
        gamma: discounted factor. If ``n_steps=1``, this value does not make
            any difference.

    Returns:
        Kiox object.

    """
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
