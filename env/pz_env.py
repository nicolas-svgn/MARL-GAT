# """CHANGE CUSTOM ENV IMPORT HERE""" ##################################################################################
from .custom_env import SUMO_PARAMS, RLController
########################################################################################################################

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from gymnasium.utils.ezpickle import EzPickle
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional


class CustomPZEnv(ParallelEnv, EzPickle):
    """
    A custom PettingZoo parallel environment for multi-agent reinforcement learning
    with traffic light control.
    
    This environment wraps the RLController to provide a standard interface
    compatible with multi-agent reinforcement learning algorithms.
    """
    
    metadata = {
        "render_modes": ["human"],
        "name": "traffic_control_v0",
    }
    
    def __init__(
        self, 
        render_mode: Optional[str] = None, 
        gui: bool = False, 
        log: bool = False, 
        rnd: Tuple[bool, bool] = (True, True)
    ):
        """
        Initialize the PettingZoo environment.
        
        Args:
            render_mode: Mode for rendering the environment
            gui: Whether to use SUMO GUI
            log: Whether to log simulation data
            rnd: Random settings for the simulation
        """
        EzPickle.__init__(
            self,
            render_mode=render_mode,
            gui=gui,
            log=log,
            rnd=rnd
        )
        
        # Initialize the SUMO environment using the RLController
        self.sumo_env = RLController(gui=gui, log=log, rnd=rnd)
        
        # Define environment properties
        self.agents = self.sumo_env.tl_ids
        self.possible_agents = self.agents[:]
        
        # Get action and observation spaces from RLController
        self.action_space_n = self.sumo_env.action_space_n
        self.observation_shape = self.sumo_env.observation_space_n

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
        
        # Store render mode
        self.render_mode = render_mode
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Random seed for the environment
            options: Optional configurations for the reset
            
        Returns:
            observations: Initial observations for each agent
            infos: Additional information for each agent
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the underlying SUMO environment
        self.sumo_env.reset()
        
        # Get initial observations for each agent
        observations = {agent: self.sumo_env.obs(agent) for agent in self.agents}
        
        # Initialize infos
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions
            
        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            terminations: Whether each agent has terminated
            truncations: Whether each agent has been truncated
            infos: Additional information for each agent
        """
        # Check if all required agents are in the actions dictionary
        if not all(agent in actions for agent in self.agents):
            missing = [agent for agent in self.agents if agent not in actions]
            raise ValueError(f"Missing actions for agents: {missing}")
        
        #Increase step of the episode
        self.current_step += 1
        
        # Execute actions in the SUMO environment
        self.sumo_env.step(actions)
        
        # Get observations and rewards for each agent
        observations = {agent: self.sumo_env.obs(agent) for agent in self.agents}
        rewards = {agent: self.sumo_env.rew(agent) for agent in self.agents}
        
        # Check if the simulation is done
        terminated = self.sumo_env.terminated()
        truncated = self.sumo_env.truncated()
        
        # Create dictionaries for terminations and truncations
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        
        # Get additional information for each agent
        infos = {agent: self.sumo_env.info_tl_id(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def state(self) -> np.ndarray:
        """Returns the global state of the environment.

        This state is a global view of the environment suitable for centralized training.
        It is formed by stacking the observations of all individual agents in a predefined order.

        Returns:
            np.ndarray: A NumPy array representing the global state.
        """
        if not self.agents:
            # If there are no agents, return an empty array with the correct
            # subsequent dimensions based on self.observation_shape.
            return np.empty((0, *self.observation_shape), dtype=np.float32)

        individual_observations = []
        # Iterate in the order of self.agents 
        for agent_id in self.agents:
            obs = self.sumo_env.obs(agent_id)  # Get individual agent observation
            individual_observations.append(obs)
        
        # Stack the observations along a new first dimension (the agent dimension)
        global_state_array = np.stack(individual_observations, axis=0)
        
        return global_state_array

    
    def render(self) -> None:
        """
        Render the environment.
        
        Note: The SUMO GUI parameter provides visual rendering,
        so this method doesn't need to do anything special.
        """
        pass
    
    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        # Stop the SUMO simulation
        self.sumo_env.stop()
    
    def observation_space(self, agent: str) -> gym.spaces.Space:
        """
        Returns the observation space for the specified agent.
        
        Args:
            agent: The agent whose observation space to return
            
        Returns:
            The observation space for the specified agent
        """
        # Use the observation shape from the SUMO environment
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_shape,
            dtype=np.float32
        )
    
    def action_space(self, agent: str) -> gym.spaces.Space:
        """
        Returns the action space for the specified agent.
        
        Args:
            agent: The agent whose action space to return
            
        Returns:
            The action space for the specified agent
        """
        # Discrete action space with action_space_n actions
        return gym.spaces.Discrete(self.action_space_n)
