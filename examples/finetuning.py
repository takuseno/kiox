import numpy as np


from kiox.offline import create_simple_kiox_from_dataset
from kiox.shortcut import create_simple_kiox


def main():
    # prepare dataset
    observations = np.random.random((1000, 10))
    actions = np.random.random((1000, 4))
    rewards = np.random.random(1000)
    terminals = np.zeros(1000)

    # setup offline Kiox
    offline_kiox = create_simple_kiox_from_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    # setup online Kiox
    online_kiox = create_simple_kiox(maxlen=1000)

    # transfer data
    online_kiox.copy_from(offline_kiox)

    # get buffer size
    print(online_kiox.get_step_buffer_size())

    # sample mini-batch
    print(online_kiox.sample(batch_size=8))


if __name__ == "__main__":
    main()
