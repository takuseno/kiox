from .step_buffer import EpisodicStepBuffer, Step, StepBuffer
from .transition_buffer import TransitionBuffer
from .transition_factory import TransitionFactory
from .types import Action, Observation


class StepCollector:
    _step_buffer: StepBuffer
    _transition_buffer: TransitionBuffer
    _transition_factory: TransitionFactory
    _episode_steps: EpisodicStepBuffer
    _n_steps: int
    _gamma: float

    def __init__(
        self,
        step_buffer: StepBuffer,
        transition_buffer: TransitionBuffer,
        transition_factory: TransitionFactory,
        n_steps: int = 1,
        gamma: float = 0.99,
        idx_offset: int = 0,
    ):
        self._step_buffer = step_buffer
        self._transition_buffer = transition_buffer
        self._transition_factory = transition_factory
        self._n_steps = n_steps
        self._gamma = gamma
        self._episode_steps = EpisodicStepBuffer()
        self._idx = idx_offset

    def collect(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        terminal: float,
    ) -> None:
        step = Step(
            idx=self._idx,
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
        )
        self._idx += 1
        self._step_buffer.append(step)
        self._episode_steps.append(step)

        if self._episode_steps.size() > self._n_steps:
            last_step = self._episode_steps.get_prev(step.idx, self._n_steps)
            assert last_step
            transition = self._transition_factory.create(
                step=last_step,
                next_step=step,
                episode_steps=self._episode_steps,
                duration=self._n_steps,
                gamma=self._gamma,
            )
            self._transition_buffer.append(transition)

        if terminal:
            # consume remaining steps
            for i in reversed(range(self._n_steps)):
                last_step = self._episode_steps.get_prev(step.idx, i)
                assert last_step
                transition = self._transition_factory.create(
                    step=last_step,
                    next_step=None,
                    episode_steps=self._episode_steps,
                    duration=i + 1,
                    gamma=self._gamma,
                )
                self._transition_buffer.append(transition)

            self.clip_episode()

    def clip_episode(self) -> None:
        self._episode_steps.clear()
