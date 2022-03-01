from kiox.step import StepBuffer

from .utility import StepFactory


def test_step_buffer():
    factory = StepFactory()
    buffer = StepBuffer()

    step1 = factory()
    buffer.append(step1)
    assert buffer.size() == 1

    step2 = factory()
    buffer.append(step2)
    assert buffer.size() == 2

    # test get
    assert buffer.get(step1.idx) is step1

    # test drop
    buffer.drop(step1.idx)
    assert buffer.size() == 1
