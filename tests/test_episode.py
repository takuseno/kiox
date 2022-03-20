from kiox.episode import Episode, EpisodeManager
from kiox.step import StepBuffer
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory

from .utility import StepFactory, TransitionFactory


def test_episode():
    factory = StepFactory()
    transition_factory = SimpleTransitionFactory()
    episode = Episode(StepBuffer(), FIFOTransitionBuffer(10))
    steps = []

    prev_step = None
    for i in range(20):
        partial_step = factory()
        step = episode.append_step(partial_step)
        steps.append(step)

        if prev_step:
            transition = transition_factory.create(
                prev_step, step, episode, 1, 0.99
            )
            dropped_transition = episode.append_transition(transition)
            if i > 10:
                assert dropped_transition is not None
            else:
                assert dropped_transition is None

        assert episode.size() == i + 1

        prev_step = step

    # test get methods
    assert episode.get(steps[0].idx) is steps[0]
    assert episode.get_by_index(0) is steps[0]
    assert episode.get_next(steps[0].idx, 1) is steps[1]
    assert episode.get_next(steps[0].idx, 2) is steps[2]
    assert episode.get_next(steps[0].idx, 20) is None
    assert episode.get_prev(steps[-1].idx, 1) is steps[-2]
    assert episode.get_prev(steps[-1].idx, 2) is steps[-3]
    assert episode.get_prev(steps[-1].idx, 20) is None

    # test compute_return
    ret = 0
    for i, step in enumerate(steps[:3]):
        ret += (0.99**i) * step.reward
    assert episode.compute_return(steps[0].idx, 3, 0.99) == ret


def test_episode_manager():
    factory = StepFactory()
    transition_factory = SimpleTransitionFactory()
    step_buffer = StepBuffer()
    episode_manager = EpisodeManager(step_buffer, FIFOTransitionBuffer(10))
    steps = []

    prev_step = None
    for i in range(11):
        step = episode_manager.append_step(factory())
        steps.append(step)

        if prev_step:
            transition = transition_factory.create(
                prev_step, step, episode_manager.active_episode, 1, 0.99
            )
            episode_manager.append_transition(transition)

        assert episode_manager.active_episode.size() == i + 1

        prev_step = step

    # test get
    assert episode_manager.get_step_by_idx(steps[0].idx) is steps[0]

    # test clip_episode
    prev_episode = episode_manager.active_episode
    episode_manager.clip_episode()
    assert episode_manager.active_episode is not prev_episode

    # test drop
    prev_step = None
    for i in range(11):
        step = episode_manager.append_step(factory())
        if prev_step:
            transition = transition_factory.create(
                prev_step, step, episode_manager.active_episode, 1, 0.99
            )
            episode_manager.append_transition(transition)
        prev_step = step

    assert len(episode_manager.episodes) == 1
    assert step_buffer.size() == 11
