# """CHANGE CUSTOM ENV IMPORT HERE""" ##################################################################################
from .custom_env import SUMO_PARAMS, RLController
########################################################################################################################

from copy import copy

import numpy as np
from gymnasium import spaces

from pettingzoo import ParallelEnv
from gymnasium.utils import EzPickle, seeding


class CustomPZEnv(ParallelEnv, EzPickle):
    """Parallel environment class.

    It steps every live agent at once. The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        'render_modes': ['ansi'],  # Supported rendering modes
        "name": "custom_environment_v0",  # Name of the environment
    }

    def __init__(self, n_agents: int = 9, render_mode='ansi'):
        """Initializes the CustomPZEnv environment.

        Args:
            n_agents (int): Number of agents in the environment (default is 9).
            render_mode (str): The rendering mode (default is 'ansi').
        """
        EzPickle.__init__(self, n_agents, render_mode)

        self.n_agents = n_agents
        self.render_mode = render_mode
        self._episode_ended = False  # Flag to track if the episode has ended

        # Initialize the SUMO environment controller
        self.sumo_env = RLController(gui=False, log=False, rnd=(True, True))

        # Get the possible agents (traffic light IDs) from the SUMO environment
        self.possible_agents = self.sumo_env.tl_ids
        self.agents = self.possible_agents[:]

        # Define the observation and action spaces for each agent
        self.observation_spaces = spaces.Dict({
            agent: spaces.Box(low=0, high=1, shape=self.sumo_env.observation_space_n, dtype=np.float32)
            for agent in self.agents
        })

        self.action_spaces = spaces.Dict({
            agent: spaces.Discrete(self.sumo_env.action_space_n)
            for agent in self.agents
        })

        self.current_step = 0  # Track the current step in the episode

    def seed(self, seed=None):
        """Sets the seed for the environment's random number generator."""
        self.np_random, seed = seeding.np_random(seed)

    def observation_space(self, agent):
        """Returns the observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Returns the action space for a given agent."""
        return self.action_spaces[agent]

    def observation(self, agent):
        """Returns the current observation for a given agent."""
        observation = self.sumo_env.obs(agent)
        return observation

    def reward(self, agent):
        """Returns the reward for a given agent."""
        reward = self.sumo_env.rew(agent)
        return reward

    def terminated(self):
        """Checks if the episode has terminated."""
        terminated = self.sumo_env.terminated()
        return terminated

    def truncated(self):
        """Checks if the episode has been truncated."""
        truncated = self.sumo_env.truncated()
        return truncated

    def info(self, agent):
        """Returns additional information for a given agent."""
        info = self.sumo_env.info_tl_id(agent)
        return info

    def reset(self, seed=None, options=None):
        """Resets the environment and returns initial observations and info.

        Args:
            seed (optional): Seed for random number generators.
            options (optional): Additional reset options.

        Returns:
            observations (dict): Initial observations for each agent.
            infos (dict): Additional information for each agent.
        """
        if seed is not None:
            self.seed(seed)

        self.sumo_env.reset()
        self.agents = self.possible_agents[:]
        self._episode_ended = False
        self.current_step = 0

        observations = {agent: self.observation(agent) for agent in self.agents}
        infos = {agent: self.info(agent) for agent in self.agents}

        return observations, {i: {} for i in self.possible_agents}

    def state(self):
        """Returns the global state of the environment.

        This state is a global view of the environment suitable for centralized training.

        Returns:
            global_state_array (np.array): Global state array.
        """
        global_state = []
        for agent in self.agents:
            global_state.append(self.sumo_env.get_dtse_array(agent))
        global_state_array = np.array(global_state)

        return global_state_array

    def step(self, actions):
        """Performs a step in the environment based on the given actions.

        Args:
            actions (dict): Actions for each agent.

        Returns:
            observations (dict): Observations for each agent.
            rewards (dict): Rewards for each agent.
            terminated (dict): Whether each agent is in a terminated state.
            truncated (dict): Whether each agent is in a truncated state.
            infos (dict): Additional information for each agent.
        """
        self.sumo_env.step(actions)
        observations = {agent: self.observation(agent) for agent in self.agents}
        rewards = {agent: self.reward(agent) for agent in self.agents}
        self.current_step += 1
        infos = {agent: self.info(agent) for agent in self.agents}
        terminated = {agent: self.terminated() for agent in self.agents}
        truncated = {agent: self.truncated() for agent in self.agents}

        return observations, rewards, terminated, truncated, infos

    def render(self, mode="human"):
        """Renders the environment.

        Args:
            mode (str): The rendering mode (default is 'human').
        """
        # TODO: IMPLEMENT
        print("TO BE IMPLEMENTED")
