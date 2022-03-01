import numpy as np
import pytest

from kiox.episode import EpisodeManager
from kiox.step import StepBuffer
from kiox.step_collector import StepCollector
from kiox.transition_buffer import UnlimitedTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


@pytest.mark.parametrize("n_steps", [1, 3])
def test_step_collector(n_steps):
    episode_manager = EpisodeManager(StepBuffer())
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    step_collector = StepCollector(
        episode_manager=episode_manager,
        transition_buffer=transition_buffer,
        transition_factory=transition_factory,
        n_steps=n_steps,
    )

    for i in range(9):
        observation = np.random.random(100)
        action = np.random.random(4)
        reward = np.random.random()
        step_collector.collect(
            observation=observation,
            action=action,
            reward=reward,
            terminal=0.0,
        )
        assert episode_manager.active_episode.size() == i + 1
        if i > n_steps:
            assert transition_buffer.size() == i + 1 - n_steps
            transition = transition_buffer.get_by_index(i - n_steps)
            step = episode_manager.active_episode.get_by_index(i - n_steps)
            next_step = episode_manager.active_episode.get_by_index(i)
            assert transition.curr_idx == step.idx
            assert transition.next_idx == next_step.idx
            assert transition.duration == n_steps

    step_collector.collect(
        observation=np.random.random(100),
        action=np.random.random(4),
        reward=np.random.random(),
        terminal=1.0,
    )

    assert episode_manager.get_total_step_size() == 10
    assert transition_buffer.size() == 10

    for i in range(n_steps):
        transition = transition_buffer.get_by_index(10 - i - 1)
        step = episode_manager.episodes[0].get_by_index(10 - i - 1)
        assert transition.curr_idx == step.idx
        assert transition.next_idx is None
        assert transition.duration == i + 1
