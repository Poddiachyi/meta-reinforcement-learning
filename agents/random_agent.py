import numpy as np
from dm_env import specs

class RandomAgent:
    """Universal random agent for DeepMind Alchemy environment both symbolic and 3d"""

    def __init__(self, action_spec):
        self.action_spec = action_spec
        self.samplying_func = self._symbolic_act if isinstance(action_spec, specs.BoundedArray) else self._env3d_act

    def act(self, obs):
        return self.samplying_func()

    def _symbolic_act(self):
        return self.action_spec.generate_value()

    def _env3d_act(self):
        action = {}

        for name, spec in self.action_spec.items():
          # Uniformly sample BoundedArray actions.
          if isinstance(spec, specs.BoundedArray):
            action[name] = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
          else:
            action[name] = spec.generate_value()
        return action
