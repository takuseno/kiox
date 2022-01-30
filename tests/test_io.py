import io
import os

from kiox.io import dump_memory, load_memory
from kiox.step_buffer import UnlimitedStepBuffer
from kiox.step_collector import StepCollector
from kiox.transition_buffer import UnlimitedTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory

from .utility import StepFactory


def test_dump_memory_and_load_memory():
    factory = StepFactory()
    buffer = UnlimitedStepBuffer()

    for _ in range(9):
        buffer.append(factory())
    buffer.append(factory(terminal=True))

    # test dump_memory
    io_byte = io.BytesIO()
    dump_memory(io_byte, buffer)

    step_buffer = UnlimitedStepBuffer()
    transition_buffer = UnlimitedTransitionBuffer()
    transition_factory = SimpleTransitionFactory()
    step_collector = StepCollector(
        step_buffer=step_buffer,
        transition_buffer=transition_buffer,
        transition_factory=transition_factory,
    )

    # test load_memory
    load_memory(io_byte, step_collector)
    assert step_buffer.size() == 10
    assert transition_buffer.size() == 10
