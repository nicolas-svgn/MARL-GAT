from env import RLController, SumoEnv, CustomEnv, network_config, GraphConverter
from agent import HistoricalBuffer, Network, BaseNetwork, GATNetwork, LSTMNetwork, ActorNetwork, CriticNetwork, TGATA2CAgent, Embedder, GATBlock

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
        print("The edge matrix of the graph is: ", self.edge_index_matrix)
        self.env = CustomEnv()
        self.tls = self.env.sumo_env.tl_ids
        print("The traffic light IDs are:", self.tls)
        self.tls_mapping = create_tl_id_mapping(self.tls)
        print("The mapping is: ",self.tls_mapping)
        self.env = ss.pettingzoo_env_to_vec_env_v1(self.env)
        self.env = ss.concat_vec_envs_v1(self.env, 1, base_class="stable_baselines3")
        self.env.reset()
        self.env_action_space = self.env.action_space
        print("The action space if of dimension: ",self.env_action_space)
        self.env_observation_space = self.env.observation_space
        print("The observation space is of dimension: ", self.env_observation_space)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print("The device is: ", self.device)
        self.input_dim = self.env_observation_space.shape
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
            agent_obs = obs[agent.tl_map_id].copy()
            agent_obs_t = self.embedder.embed_agent_obs(agent_obs)
            agent.load_hist_buffer(agent_obs_t)

    def train_loop(self):
        print()
        print("Start Training")

        # Enable anomaly detection
        T.autograd.set_detect_anomaly(True)  

        """for step in itertools.count(start=self.agent.resume_step):
            self.agent.step = step"""

        actions = [random.randint(0,3) for tl_id in self.tls]
        obs, rew, terminated, infos = self.env.step(actions)

        graph_features = self.embedder.graph_embed_state(obs)

        gat_output = self.gat_block.gat_output(graph_features)

        for agent in self.agents:
           agent.gat_features = gat_output.clone()
           agent_obs = obs[agent.tl_map_id].copy()
           embedded_agent_obs = self.embedder.embed_agent_obs(agent_obs)
           agent.current_t_obs = embedded_agent_obs.clone()

        for step in range(3):

            actions = []
            agent_log_probs = []

            for agent in self.agents:
                action, log_prob = agent.select_action(agent.current_t_obs, agent.gat_features)
                agent.current_action = action
                actions.append(agent.current_action)
                agent_log_probs.append(log_prob)

            new_obs, rew, terminated, infos = self.env.step(actions)
            new_graph_features = self.embedder.graph_embed_state(new_obs)
            new_gat_output = self.gat_block.gat_output(new_graph_features)

            for agent in self.agents:
                agent.new_gat_features = new_gat_output.clone()
                agent_new_obs = new_obs[agent.tl_map_id].copy()
                embedded_agent_new_obs = self.embedder.embed_agent_obs(agent_new_obs)
                agent.new_t_obs = embedded_agent_new_obs.clone()


            vlosses = []
            plosses = []

            for agent in self.agents:
                print('--------------------')
                print('agent id')
                print(agent.tl_id)
                print('agent map id')
                print(agent.tl_map_id)
                agent_action = agent.current_action
                agent_action_log_prob = agent_log_probs[agent.tl_map_id]
                print('agent action')
                print(agent_action)
                agent_reward = rew[agent.tl_map_id]
                print('agent reward')
                print(agent_reward)
                agent_terminated = terminated[agent.tl_map_id]
                print('agent is done ?')
                print(agent_terminated)
                print('--------------------')

                vloss, ploss = agent.learn(agent.gat_features, agent.new_gat_features, agent_action_log_prob, agent.current_t_obs, agent.new_t_obs, agent_reward, agent_terminated)
                vlosses.append(vloss)
                plosses.append(ploss)

            # Calculate the average losses across all agents
            avg_value_loss = sum(vlosses) / len(vlosses)
            avg_policy_loss = sum(plosses) / len(plosses)

            # Combine the average losses
            total_loss = avg_value_loss + avg_policy_loss

            # Zero gradients for all optimizers (shared and individual)
            self.embedder.base_network.optimizer.zero_grad()
            self.gat_block.gat_network.optimizer.zero_grad()
            for agent in self.agents:
                agent.lstm_network.optimizer.zero_grad()
                agent.actor_network.optimizer.zero_grad()
                agent.critic_network.optimizer.zero_grad()

            # Disable dropout for backpropagation
            self.gat_block.gat_network.train(False)

            # Backpropagate the total loss only once
            print('we re about to backward')
            total_loss.backward(retain_graph=True)
            print('backward done !')

            # Check gradients for the BaseNetwork
            for name, param in self.embedder.base_network.named_parameters():
                if param.grad is not None:
                    print(f"Gradient computed for {name}")
                else:
                    print(f"No gradient computed for {name}")

            # Re-enable dropout
            self.gat_block.gat_network.train(True)

            # Update all optimizers (shared and individual)
            self.embedder.base_network.optimizer.step()
            self.gat_block.gat_network.optimizer.step()
            for agent in self.agents:
                agent.lstm_network.optimizer.step()
                agent.actor_network.optimizer.step()
                agent.critic_network.optimizer.step()

            for agent in self.agents:
                agent.load_hist_buffer(agent.current_t_obs)
                agent.gat_features = agent.new_gat_features.clone()
                agent.current_t_obs = agent.new_t_obs.clone()









    def run(self):
       self.load_agent_buffers()
       self.train_loop()



if __name__ == "__main__":
   train = Train()
   train.run()