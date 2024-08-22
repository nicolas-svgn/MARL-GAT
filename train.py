from env import RLController, SumoEnv, CustomEnv, network_config, GraphConverter
from agent import HistoricalBuffer, Network, MLPNetwork, BaseNetwork, GATNetwork, LSTMNetwork, ActorNetwork, CriticNetwork, TGATA2CAgent, Embedder, GATBlock

import random

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch import no_grad, as_tensor

import os
import sys
import time
import argparse
import itertools
from datetime import timedelta
import numpy as np
import supersuit as ss
from stable_baselines3.common.env_checker import check_env
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

def create_tl_id_mapping(tls):
  """Creates a mapping from tl_ids to integer indices.

  Args:
      tls: A list of tl_ids in the desired order.

  Returns:
      A dictionary mapping tl_ids to their corresponding integer indices.
  """
  return {tl_id: i for i, tl_id in enumerate(tls)}

class Train:
    def __init__(self):
        self.file_path = "plain.edg.csv"
        self.converter = GraphConverter(self.file_path) 
        self.graph = self.converter.create_graph() 
        #self.converter.visualize_graph(self.graph)
        self.edge_index_matrix = self.converter.get_edge_matrix(self.graph)
        print(self.edge_index_matrix)
        self.env = CustomEnv()
        self.tls = self.env.sumo_env.tl_ids
        self.tls_mapping = create_tl_id_mapping(self.tls)
        print(self.tls_mapping)
        self.env = ss.pettingzoo_env_to_vec_env_v1(self.env)
        self.env = ss.concat_vec_envs_v1(self.env, 1, base_class="stable_baselines3")
        self.env.reset()
        self.env_action_space = self.env.action_space
        print(self.env_action_space)
        self.env_observation_space = self.env.observation_space
        print(self.env_observation_space)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.input_dim = self.env_observation_space
        self.lr = 0.001
        self.gamma = 0.99
        self.entropy_weight = 1e-2
        self.agents = []
        self.embedder = Embedder(self.device, self.input_dim, self.lr)
        self.gat_block = GATBlock(self.device, self.lr, self.edge_index_matrix)

        for tl_id in self.tls:
           agent = TGATA2CAgent(device = self.device, input_dim = self.input_dim, num_actions = self.env_action_space, lr_base = self.lr, lr_gat = self.lr, lr_lstm = self.lr, lr_actor = self.lr, lr_critic = self.lr, gamma = self.gamma, 
                 entropy_weight = self.entropy_weight, epsilon_start = 1., epsilon_min = 0.01, epsilon_decay = 2e6, epsilon_exp_decay = True, 
                 save_frequency = 10000, log_frequency = 1000, save_dir = '', log_dir = '', load = True, tl_id = tl_id, historical_buffer_size=9, edge_index=self.edge_index_matrix)
           self.agents.append(agent)



    def load_agent_buffers(self):
       for _ in range(10):
        actions = [random.randint(0,3) for tl_id in self.tls]
        obs, rew, terminated, infos = self.env.step(actions)
        for agent in self.agents:
            obs_tensor = self.embedder.embed_agent_obs(obs[agent.tl_map_id])
            agent.load_hist_buffer(obs_tensor)

    def train_loop(self):
        print()
        print("Start Training")

        """for step in itertools.count(start=self.agent.resume_step):
            self.agent.step = step"""

        actions = [random.randint(0,3) for tl_id in self.tls]
        obs, rew, terminated, infos = self.env.step(actions)

        graph_features = self.embedder.graph_embed_state(obs)
        print('graph features')
        #print(graph_features)

        gat_output = self.gat_block.gat_output(graph_features)
        print('gat output')
        #print(gat_output)

        gnej1_obs = self.embedder.embed_agent_obs(obs[0])

        """combined_output = self.agents[0].temporal_graph_forward(gnej1_obs, gat_output)
        print('combined output')
        print(combined_output)"""
        
        #actions = {agent.tl_id : agent.select_action(obs) for agent in self.agents}
        actions = [agent.select_action(self.embedder.embed_agent_obs(obs[agent.tl_map_id]), gat_output)[0] for agent in self.agents]
        print(actions)
        new_obs, rew, terminated, infos = self.env.step(actions)
        new_graph_features = self.embedder.graph_embed_state(new_obs)
        new_gat_output = self.gat_block.gat_output(new_graph_features)
        for agent in self.agents:
           print('--------------------')
           print('agent id')
           print(agent.tl_id)
           agent_map_id = self.tls_mapping[agent.tl_id]
           print('agent map id')
           #print(agent_map_id)
           #print(agent.tl_map_id)
           agent_obs = self.embedder.embed_agent_obs(obs[agent_map_id])
           print('agent obs')
           print(agent_obs)
           agent_new_obs = self.embedder.embed_agent_obs(new_obs[agent_map_id])
           print('agent new obs')
           print(agent_new_obs)
           agent_action, agent_action_log_probs = agent.select_action(agent_obs, gat_output)
           print('agent action')
           print(agent_action)
           if agent_action_log_probs is not None:
            print('agent action log prob')
            print(agent_action_log_probs[agent_action])
           agent_reward = rew[agent_map_id]
           print('agent reward')
           print(agent_reward)
           agent_terminated = terminated[agent_map_id]
           print('agent is done ?')
           print(agent_terminated)
           print('--------------------')
           agent.learn(gat_output, new_gat_output, 0.5, agent_obs, agent_new_obs, agent_reward, agent_terminated)

        obs = new_obs



    def run(self):
       self.load_agent_buffers()
       self.train_loop()
       """for agent in self.agents:
           print('------------------------------')
           print(agent.tl_id)
           print(agent.historical_buffer.get())
           print('------------------------------')"""



if __name__ == "__main__":
   train = Train()
   train.run()


