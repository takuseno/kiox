from multiprocessing import Process

import gym
import numpy as np

from kiox.distributed.server import KioxServer
from kiox.distributed.step_sender import StepSender
from kiox.step_buffer import FIFOStepBuffer
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def rollout():
    sender = StepSender("localhost", 8000, 1)
    env = gym.make("CartPole-v0")

    obs = env.reset()
    while True:
        action = np.random.randint(2)
        next_obs, reward, terminal, _ = env.step(action)
        sender.collect(obs.astype(np.float32), action, reward, terminal)
        if terminal:
            break
        obs = next_obs
    sender.stop()


def main():
    def step_buffer_builder():
        return FIFOStepBuffer(1000)

    def transition_buffer_builder():
        return FIFOTransitionBuffer(1000)

    def transition_factory_builder():
        return SimpleTransitionFactory()

    # setup server
    server = KioxServer(
        host="localhost",
        port=8000,
        observation_shape=(4,),
        action_shape=(1,),
        reward_shape=(1,),
        batch_size=8,
        step_buffer_builder=step_buffer_builder,
        transition_buffer_builder=transition_buffer_builder,
        transition_factory_builder=transition_factory_builder,
    )
    server.start()

    # start rollout
    p = Process(target=rollout)
    p.start()
    p.join()

    print(server.get_step_buffer_size())
    print(server.sample())

    server.stop()


if __name__ == "__main__":
    main()
