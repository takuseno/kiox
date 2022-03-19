from kiox.step import StepBuffer

from .utility import StepFactory


def test_step_buffer():
    factory = StepFactory()
    buffer = StepBuffer()

    partial_step1 = factory()
    step1 = buffer.append(partial_step1)
    assert buffer.size() == 1
    assert step1.idx == 0

    partial_step2 = factory()
    step2 = buffer.append(partial_step2)
    assert buffer.size() == 2
    assert step2.idx == 1

    # test get
    assert buffer.get(step1.idx) is step1

    # test drop
    buffer.drop(step1.idx)
    assert buffer.size() == 1
