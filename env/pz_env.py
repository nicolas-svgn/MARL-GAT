# """CHANGE CUSTOM ENV IMPORT HERE""" ##################################################################################
from .custom_env import SUMO_PARAMS, RLController
########################################################################################################################

import functools
import random
from copy import copy

import numpy as np
from gymnasium import spaces

from pettingzoo import ParallelEnv
from gymnasium.utils import EzPickle, seeding


class CustomPZEnv(ParallelEnv, EzPickle):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, n_agents: int = 9):
        """The init method takes in environment arguments.

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        EzPickle.__init__(
            self,
            n_agents
        )

        self.n_agents = n_agents

        self._episode_ended = False

        self.sumo_env = RLController(gui=SUMO_PARAMS["gui"], log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"])

        self.possible_agents = self.sumo_env.tl_ids

        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.agents))))
        )

        self.observation_spaces = spaces.Dict({agent: spaces.Box(low=0, high=1, shape=self.sumo_env.observation_space_n, dtype=np.float32) for agent in self.agents})

        self.action_spaces = spaces.Dict(
            {agent: spaces.Discrete(self.sumo_env.action_space_n) for agent in self.agents}
        )

        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.sumo_env.reset()

        self.agents = self.possible_agents[:]
        self._episode_ended = False
        self.current_step = 0

    def state(self):
        global_state = []
        for agent in self.agents:
            global_state.append(self.sumo_env.get_dtse_array(agent))
            global_state_array = np.array(global_state)

        return global_state_array
    
