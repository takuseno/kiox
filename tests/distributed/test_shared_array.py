from multiprocessing import Process

import numpy as np

from kiox.distributed.shared_array import create_shared_array


def test_create_shared_array():
    array = create_shared_array([3, 84, 84], dtype=np.float32)

    def child(array):
        array.fill(1.0)

    p = Process(target=child, args=(array,))
    p.start()
    p.join()

    assert np.all(array == 1.0)
