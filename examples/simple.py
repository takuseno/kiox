import gym
import numpy as np


from kiox.kiox import Kiox
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory


def main():
    # setup Kiox
    kiox = Kiox(FIFOTransitionBuffer(1000), SimpleTransitionFactory())

    # collect data
    env = gym.make("CartPole-v0")
    obs = env.reset()
    while True:
        action = np.random.randint(2)
        next_obs, reward, terminal, _ = env.step(action)
        kiox.collect(obs, action, reward, terminal)
        if terminal:
            break
        obs = next_obs

    # get buffer size
    print(kiox.get_step_buffer_size())

    # sample mini-batch
    print(kiox.sample(batch_size=8))


if __name__ == "__main__":
    main()
