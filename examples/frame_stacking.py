import numpy as np


from kiox.kiox import Kiox
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import FrameStackTransitionFactory


def main():
    # setup Kiox
    kiox = Kiox(FIFOTransitionBuffer(1000), FrameStackTransitionFactory(4))

    # collect data
    for i in range(1000):
        observation = np.random.random((3, 84, 84))
        action = np.random.randint(2)
        reward = np.random.random()
        terminal = (i % 100) == 0
        kiox.collect(observation, action, reward, terminal)

    # get buffer size
    print(kiox.get_step_buffer_size())

    # sample mini-batch
    batch = kiox.sample(batch_size=8)
    print(batch.observations.shape)


if __name__ == "__main__":
    main()
