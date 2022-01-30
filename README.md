# kiox: A composable experience replay buffer library
[![test](https://github.com/takuseno/kiox/actions/workflows/test.yml/badge.svg)](https://github.com/takuseno/kiox/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/takuseno/kiox/branch/master/graph/badge.svg?token=sI8AYW2kYp)](https://codecov.io/gh/takuseno/kiox)
[![Maintainability](https://api.codeclimate.com/v1/badges/f2f0d2bde462dbb37767/maintainability)](https://codeclimate.com/github/takuseno/kiox/maintainability)
![MIT](https://img.shields.io/badge/license-MIT-blue)

kiox is a composable experience replay buffer library.

```py
from kiox.kiox import Kiox
from kiox.step_buffer import FIFOStepBuffer
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import SimpleTransitionFactory

kiox = Kiox(FIFOStepBuffer(1000), FIFOTransitionBuffer(1000), SimpleTransitionFactory())

# collect experiences
kiox.collect(<obsrvation>, <action>, <reward>, <terminal>)

# sample batch
batch = kiox.sample(256)

# convenient shortcut
from kiox.shortcut import create_simple_kiox
kiox = create_simple_kiox(1000)

# multi-step learning
kiox = create_simple_kiox(1000, n_steps=5, gamma=0.99)

# frame stacking
from kiox.transition_factory import FrameStackTransitionFactory
kiox = Kiox(FIFOStepBuffer(1000), FIFOTransitionBuffer(1000), FrameStackTransitionFactory(n_frames=4))

# from offline data
from kiox.offline import create_simple_kiox_from_data
kiox = create_simple_kiox_from_data(
  observations=<observations>,
  actions=<actions>,
  rewards=<rewards>,
  terminals=<terminals>,
)
```

## TODO
This project is in progress.

- [ ] gRPC server/client
- [ ] Documentation
- [ ] PyPi upload
