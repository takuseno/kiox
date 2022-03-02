from multiprocessing import Process

import gym
import numpy as np

from kiox.distributed.server import KioxServer
from kiox.distributed.step_sender import StepSender
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def rollout():
    sender = StepSender("localhost", 8000, 1)
    for i in range(1000):
        observation = (
            np.random.random(100).astype(np.float32),
            np.random.random((3, 84, 84)).astype(np.float32),
        )
        action = np.random.random(4).astype(np.float32)
        reward = np.random.random()
        terminal = (i % 100) == 0
        sender.collect(observation, action, reward, terminal)
    sender.stop()


def main():
    def transition_buffer_builder():
        return FIFOTransitionBuffer(1000)

    def transition_factory_builder():
        return SimpleTransitionFactory()

    # setup server
    server = KioxServer(
        host="localhost",
        port=8000,
        observation_shape=((100,), (3, 84, 84)),
        action_shape=(4,),
        reward_shape=(1,),
        batch_size=8,
        transition_buffer_builder=transition_buffer_builder,
        transition_factory_builder=transition_factory_builder,
    )
    server.start()

    # start rollout
    p = Process(target=rollout)
    p.start()

    # wait until episode ends
    p.join()

    # sample mini-batch
    batch = server.sample()
    print(batch.observations[0].shape)
    print(batch.observations[1].shape)

    server.stop()


if __name__ == "__main__":
    main()
