import numpy as np


from kiox.offline import create_simple_kiox_from_dataset


def main():
    # prepare dataset
    observations = np.random.random((1000, 10))
    actions = np.random.random((1000, 4))
    rewards = np.random.random(1000)
    terminals = np.zeros(1000)

    # setup Kiox
    kiox = create_simple_kiox_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    # get buffer size
    print(kiox.get_step_buffer_size())

    # sample mini-batch
    print(kiox.sample(batch_size=8))


if __name__ == "__main__":
    main()
