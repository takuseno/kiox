from kiox.step_buffer import (
    EpisodicStepBuffer,
    FIFOStepBuffer,
    UnlimitedStepBuffer,
)

from .utility import StepFactory


def test_unlimited_step_buffer():
    factory = StepFactory()
    buffer = UnlimitedStepBuffer()
    steps = []

    for i in range(10):
        step = factory()
        buffer.append(step)
        steps.append(step)
        assert buffer.size() == i + 1

    # test get methods
    assert buffer.get(steps[0].idx) is steps[0]
    assert buffer.get_by_index(0) is steps[0]

    # test copy_from
    buffer2 = UnlimitedStepBuffer()
    buffer2.copy_from(buffer)
    assert buffer2.size() == 10
    assert buffer2.get(steps[0].idx) is steps[0]

    # test clear
    buffer.clear()
    assert buffer.size() == 0


def test_fifo_step_buffer():
    factory = StepFactory()
    buffer = FIFOStepBuffer(5)
    steps = []

    for i in range(10):
        step = factory()
        buffer.append(step)
        steps.append(step)
        assert buffer.size() == min(i + 1, 5)

    # test get methods
    assert buffer.get(steps[5].idx) is steps[5]
    assert buffer.get_by_index(0) is steps[5]

    # test copy_from
    buffer2 = FIFOStepBuffer(5)
    buffer2.copy_from(buffer)
    assert buffer2.size() == 5
    assert buffer2.get(steps[5].idx) is steps[5]

    # test clear
    buffer.clear()
    assert buffer.size() == 0


def test_episodic_step_buffer():
    factory = StepFactory()
    buffer = EpisodicStepBuffer()
    steps = []

    for i in range(10):
        step = factory()
        buffer.append(step)
        steps.append(step)
        assert buffer.size() == i + 1

    # test get methods
    assert buffer.get(steps[0].idx) is steps[0]
    assert buffer.get_by_index(0) is steps[0]
    assert buffer.get_next(steps[0].idx, 1) is steps[1]
    assert buffer.get_next(steps[0].idx, 2) is steps[2]
    assert buffer.get_next(steps[0].idx, 10) is None
    assert buffer.get_prev(steps[-1].idx, 1) is steps[-2]
    assert buffer.get_prev(steps[-1].idx, 2) is steps[-3]
    assert buffer.get_prev(steps[-1].idx, 10) is None

    # test compute_return
    ret = 0
    for i, step in enumerate(steps[:3]):
        ret += (0.99**i) * step.reward
    assert buffer.compute_return(steps[0].idx, 3, 0.99) == ret

    # test clear
    buffer.clear()
    assert buffer.size() == 0
