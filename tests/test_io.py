import io
import os

from kiox.episode import EpisodeManager
from kiox.io import dump_memory, load_memory
from kiox.step import StepBuffer
from kiox.step_collector import StepCollector
from kiox.transition_buffer import UnlimitedTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory

from .utility import StepFactory


def test_dump_memory_and_load_memory():
    factory = StepFactory()
    episode_manager = EpisodeManager(StepBuffer(), UnlimitedTransitionBuffer())

    for _ in range(9):
        episode_manager.append_step(factory())
    episode_manager.append_step(factory(terminal=True))

    # test dump_memory
    io_byte = io.BytesIO()
    dump_memory(io_byte, episode_manager.episodes)

    transition_buffer = UnlimitedTransitionBuffer()
    episode_manager2 = EpisodeManager(StepBuffer(), transition_buffer)
    transition_factory = SimpleTransitionFactory()
    step_collector = StepCollector(
        episode_manager=episode_manager2,
        transition_factory=transition_factory,
    )

    # test load_memory
    load_memory(io_byte, step_collector)
    assert episode_manager2.get_total_step_size() == 10
    assert transition_buffer.size() == 10
