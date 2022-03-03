import dataclasses
import os

import numpy as np
from PIL import Image


from kiox.kiox import Kiox
from kiox.transition_buffer import FIFOTransitionBuffer
from kiox.transition_factory import TransitionFactory
from kiox.transition import Transition, LazyTransition


def prepare_dummy_images():
    os.makedirs("image_data", exist_ok=True)
    for i in range(100):
        image = np.random.randint(256, size=(84, 84, 3), dtype=np.uint8)
        path = os.path.join("image_data", f"{i}.png")
        Image.fromarray(image).save(path)


@dataclasses.dataclass(frozen=True)
class ImageLoadTransition(LazyTransition):
    def create(self, step_buffer):
        # load image from disk
        step = step_buffer.get(self.curr_idx)
        observation = np.asarray(Image.open(step.observation))

        # load next image from disk
        if self.next_idx is None:
            next_observation = np.zeros_like(observation)
        else:
            next_step = step_buffer.get(self.next_idx)
            next_observation = np.asarray(Image.open(next_step.observation))

        return Transition(
            observation=observation,
            action=step.action,
            reward=self.multi_step_reward,
            next_observation=next_observation,
            terminal=step.terminal,
            duration=self.duration,
        )


class ImageLoadTransitionFactory(TransitionFactory):
    def create(self, step, next_step, episode, duration, gamma):
        return ImageLoadTransition(
            curr_idx=step.idx,
            next_idx=None if next_step is None else next_step.idx,
            multi_step_reward=episode.compute_return(step.idx, duration, gamma),
            duration=duration,
        )


def main():
    # prepare dummy images
    prepare_dummy_images()

    # setup Kiox
    kiox = Kiox(FIFOTransitionBuffer(1000), ImageLoadTransitionFactory())

    # collect data
    for i in range(100):
        observation = os.path.join("image_data", f"{i}.png")
        action = np.random.randint(2)
        reward = np.random.random()
        terminal = 0
        kiox.collect(observation, action, reward, terminal)

    # get buffer size
    print(kiox.get_step_buffer_size())

    # sample mini-batch
    batch = kiox.sample(batch_size=8)
    print(batch.observations.shape)


if __name__ == "__main__":
    main()
