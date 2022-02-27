from kiox.episode import Episode, EpisodeManager

from .utility import StepFactory


def test_episode():
    factory = StepFactory()
    episode = Episode()
    steps = []

    for i in range(10):
        step = factory()
        episode.append(step)
        steps.append(step)
        assert episode.size() == i + 1

    # test get methods
    assert episode.get(steps[0].idx) is steps[0]
    assert episode.get_by_index(0) is steps[0]
    assert episode.get_next(steps[0].idx, 1) is steps[1]
    assert episode.get_next(steps[0].idx, 2) is steps[2]
    assert episode.get_next(steps[0].idx, 10) is None
    assert episode.get_prev(steps[-1].idx, 1) is steps[-2]
    assert episode.get_prev(steps[-1].idx, 2) is steps[-3]
    assert episode.get_prev(steps[-1].idx, 10) is None

    # test compute_return
    ret = 0
    for i, step in enumerate(steps[:3]):
        ret += (0.99**i) * step.reward
    assert episode.compute_return(steps[0].idx, 3, 0.99) == ret


def test_episode_manager():
    factory = StepFactory()
    episode_manager = EpisodeManager()
    steps = []

    for i in range(10):
        step = factory()
        episode_manager.append(step)
        steps.append(step)
        assert episode_manager.active_episode.size() == i + 1

    # test get
    assert episode_manager.get_step_by_idx(steps[0].idx) is steps[0]

    # test copy_from
    dst = EpisodeManager()
    dst.copy_from(episode_manager)
    assert dst.episodes[0].size() == len(steps)

    # test clip_episode
    prev_episode = episode_manager.active_episode
    episode_manager.clip_episode()
    assert episode_manager.active_episode is not prev_episode
    assert episode_manager.get_episode_by_step_idx(steps[0].idx) is prev_episode

    # test drop
    assert len(episode_manager.episodes) == 2
    for step in steps:
        episode_manager.drop_step(step.idx)
    assert len(episode_manager.episodes) == 1
