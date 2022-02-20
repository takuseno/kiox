import os
import time
from multiprocessing import Process, Queue

import numpy as np

from kiox.distributed.server import (
    COMMAND_GET_STEP_LEN,
    COMMAND_GET_TRANSITION_LEN,
    COMMAND_LOAD,
    COMMAND_SAMPLE,
    COMMAND_SAVE,
    COMMAND_STOP,
    KioxServer,
    kiox_server_process,
)
from kiox.distributed.shared_batch_factory import SharedBatchFactory
from kiox.distributed.step_sender import StepSender
from kiox.step_buffer import FIFOStepBuffer
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def test_kiox_server_process():
    def step_buffer_builder():
        return FIFOStepBuffer(10)

    def transition_buffer_builder():
        return FIFOTransitionBuffer(10)

    def transition_factory_builder():
        return SimpleTransitionFactory()

    command_queue = Queue()
    ack_queue = Queue()

    batch_factory = SharedBatchFactory((3, 84, 84), (4,), 1)

    # start server
    process = Process(
        target=kiox_server_process,
        args=(
            "localhost",
            8000,
            batch_factory,
            command_queue,
            ack_queue,
            step_buffer_builder,
            transition_buffer_builder,
            transition_factory_builder,
        ),
    )
    process.start()

    # wait until start
    ack_queue.get()

    time.sleep(1)

    sender = StepSender("localhost", 8000, 1)

    observation = np.random.random((3, 84, 84)).astype(np.float32)
    action = np.random.random(4)
    reward = np.random.random()
    sender.collect(
        observation=observation,
        action=action,
        reward=reward,
        terminal=0.0,
    )
    sender.collect(
        observation=observation,
        action=action,
        reward=reward,
        terminal=0.0,
    )

    time.sleep(2)

    # check number of steps
    command_queue.put(COMMAND_GET_STEP_LEN)
    assert int(ack_queue.get()) == 2

    # check number of transitions
    command_queue.put(COMMAND_GET_TRANSITION_LEN)
    assert int(ack_queue.get()) == 1

    assert os.path.exists("test_data"), "Please make test_data directory."

    # check save
    command_queue.put(COMMAND_SAVE)
    command_queue.put(os.path.join("test_data", "kiox.h5"))
    ack_queue.get()

    # check load
    command_queue.put(COMMAND_LOAD)
    command_queue.put(os.path.join("test_data", "kiox.h5"))
    ack_queue.get()
    command_queue.put(COMMAND_GET_STEP_LEN)
    assert int(ack_queue.get()) == 4

    # check sample
    command_queue.put(COMMAND_SAMPLE)
    ack_queue.get()
    assert np.all(batch_factory.batch.observations != 0.0)

    command_queue.put(COMMAND_STOP)

    # wait until finished
    ack_queue.get()

    sender.stop()


def test_kiox_server():
    def step_buffer_builder():
        return FIFOStepBuffer(10)

    def transition_buffer_builder():
        return FIFOTransitionBuffer(10)

    def transition_factory_builder():
        return SimpleTransitionFactory()

    server = KioxServer(
        host="localhost",
        port=8000,
        observation_shape=(3, 84, 84),
        action_shape=(4,),
        batch_size=1,
        step_buffer_builder=step_buffer_builder,
        transition_buffer_builder=transition_buffer_builder,
        transition_factory_builder=transition_factory_builder,
    )
    server.start()

    time.sleep(1)

    sender = StepSender("localhost", 8000, 1)

    observation = np.random.random((3, 84, 84)).astype(np.float32)
    action = np.random.random(4)
    reward = np.random.random()
    sender.collect(
        observation=observation,
        action=action,
        reward=reward,
        terminal=0.0,
    )
    sender.collect(
        observation=observation,
        action=action,
        reward=reward,
        terminal=0.0,
    )

    time.sleep(2)

    # check number of steps
    assert server.get_step_buffer_size() == 2

    # check number of transitions
    assert server.get_transition_buffer_size() == 1

    # check save
    server.save(os.path.join("test_data", "kiox.h5"))

    # check load
    server.load(os.path.join("test_data", "kiox.h5"))
    assert server.get_step_buffer_size() == 4

    # check sample
    batch = server.sample()
    assert batch.observations.shape == (1, 3, 84, 84)
    assert batch.actions.shape == (1, 4)
    assert batch.rewards.shape == (1, 1)
    assert batch.terminals.shape == (1, 1)
    assert np.all(batch.observations != 0.0)
    assert np.all(batch.actions != 0.0)
    assert np.all(batch.rewards != 0.0)

    server.stop()
    sender.stop()
