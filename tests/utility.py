import numpy as np

from kiox.episode import EpisodeManager
from kiox.step import Step, StepBuffer
from kiox.transition import SimpleLazyTransition


class StepFactory:
    def __init__(
        self,
        observation_shape=(100,),
        action_type="continuous",
        action_size=4,
        idx_offset=0,
    ):
        self.observation_shape = observation_shape
        self.action_type = action_type
        self.action_size = action_size
        self.idx = idx_offset

    def __call__(self, terminal=False):
        if self.action_type == "continuous":
            action = np.random.random(self.action_size)
        elif self.action_type == "discrete":
            action = np.random.randint(self.action_size)
        else:
            raise ValueError(f"invalid action type: {self.action_type}")

        if isinstance(self.observation_shape[0], int):
            observation = np.random.random(self.observation_shape)
        else:
            observation = [
                np.random.random(shape) for shape in self.observation_shape
            ]

        step = Step(
            idx=self.idx,
            observation=observation,
            action=action,
            reward=np.random.random(),
            terminal=1.0 if terminal else 0.0,
        )

        self.idx += 1

        return step

    def fill(self, n_steps):
        for _ in range(n_steps):
            self.__call__()


class TransitionFactory:
    def __init__(self, step_factory):
        self.step_factory = step_factory
        self.prev_step = step_factory()
        self.step_buffer = StepBuffer()
        self.episode_manager = EpisodeManager(self.step_buffer)
        self.episode_manager.append(self.prev_step)

    def __call__(self, terminal=False):
        step = self.step_factory()
        transition = SimpleLazyTransition(
            self.prev_step.idx,
            None if terminal else step.idx,
            multi_step_reward=np.random.random(),
            duration=1,
        )
        self.prev_step = step
        self.episode_manager.append(step)
        return transition
