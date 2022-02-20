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
```


## features
### distributed training
See an [example](examples/distributed.py).

In actor process:
```py
from kiox.distributed.step_sender import StepSender
sender = StepSender("localhost", 8000, 1)
sender.collect(<obsrvation>, <action>, <reward>, <terminal>)
```

In trainer process:
```py
# trainer process
from kiox.distributed.server import KioxServer

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
    batch_size=8,
    step_buffer_builder=step_buffer_builder,
    transition_buffer_builder=transition_buffer_builder,
    transition_factory_builder=transition_factory_builder,
)
server.start()

# sample batch
batch = server.sample()
```

### from offline data
```py
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

- [x] gRPC server/client
- [ ] Documentation
- [ ] PyPi upload
