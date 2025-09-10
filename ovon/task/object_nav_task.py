from typing import List, Optional

import os

import attr

from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask

@registry.register_task(name="ObjectNav-v2")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int
    _is_resetting: bool

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False
        self._is_resetting = False
    
    def reset(self, episode):
        self._is_resetting = True
        obs = super().reset(episode)
        self._is_resetting = False
        return obs
