import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import os
import time
import math
import random
import numpy as np
from collections import deque
from datetime import timedelta

from torch.utils.tensorboard import SummaryWriter

from .utils import ABCMeta, abstract_attribute

from .hist_buffer import HistoricalBuffer

from .network2 import BaseNetwork, GATNetwork, LSTMNetwork, ActorNetwork, CriticNetwork

tls_mapping = {'gneJ1': 0, 'gneJ10': 1, 'gneJ13': 2, 'gneJ15': 3, 'gneJ18': 4, 'gneJ3': 5, 'gneJ20': 6, 'gneJ5': 7, 'gneJ8': 8}


class TGATA2CAgent:
    def __init__(self, device, input_dim, num_actions, lr_base, lr_gat, lr_lstm, lr_actor, lr_critic, gamma, entropy_weight,
                 epsilon_start, epsilon_min, epsilon_decay, epsilon_exp_decay, 
                 save_frequency, log_frequency, save_dir, log_dir, load, tl_id, historical_buffer_size=9, edge_index=None):
        self.tl_id = tl_id
        self.tl_map_id = tls_mapping[self.tl_id]
        self.device = device
        self.gamma = gamma
        self.historical_buffer_size = historical_buffer_size
        self.entropy_weight = entropy_weight

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_exp_decay = epsilon_exp_decay

        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.load = load

        self.step = 0
        self.resume_step = 0
        self.episode_count = 0

        path = 'TGATA2C' + tl_id
        self.save_path = save_dir + path + '_' + 'model.pack'
        self.summary_writer = SummaryWriter(log_dir + path + '/')

        self.start_time = time.time()

        # Networks
        #self.gat_network = GATNetwork(lr_gat).to(device)
        self.lstm_network = LSTMNetwork(lr_lstm).to(device)
        self.actor_network = ActorNetwork(lr_actor, input_dim=4).to(device)  # Assuming combined output dim is 4
        self.critic_network = CriticNetwork(lr_critic, input_dim=4).to(device)

        # Historical buffer for LSTM
        self.historical_buffer = HistoricalBuffer(self.tl_id)
        self.edge_index = edge_index  # Store the edge index for GAT

    def load_model(self):
        if self.load and os.path.exists(self.save_path):
            print()
            print("Resume training from " + self.save_path + "...")
            self.resume_step, self.episode_count, rew_mean, len_mean = self.online_network.load(self.save_path)
            [self.ep_info_buffer.append({'r': rew_mean, 'l': len_mean}) for _ in range(np.min([self.episode_count, self.ep_info_buffer.maxlen]))]
            print("Step: ", self.resume_step * self.n_env, ", Episodes: ", self.episode_count, ", Avg Rew: ", rew_mean, ", Avg Ep Len: ", len_mean)

            self.update_target_network(force=True)
            self.step = self.resume_step

    def save_model(self):
        if self.step % self.save_frequency == 0 and self.step > self.resume_step:
            print()
            print("Saving model...")
            self.online_network.save(self.save_path, self.step, self.episode_count, self.info_mean('r'), self.info_mean('l'))
            print("OK!")

    def epsilon(self):
        if self.epsilon_exp_decay:
            return np.exp(np.interp(self.step, [0, self.epsilon_decay], [np.log(self.epsilon_start), np.log(self.epsilon_min)]))
        else:
            return np.interp(self.step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_min])
        
    def load_hist_buffer(self, obs_tensor):
        assert obs_tensor.size() == torch.Size([1, 8]), f"Expected obs_tensor size: torch.Size([1, 8]), but got {obs_tensor.size()}"
        self.historical_buffer.store(obs_tensor)


    def temporal_graph_forward(self, obs, gat_features):
        historical_obs = self.historical_buffer.get()
        historical_obs.append(obs)
        gat_agent_feature = gat_features[self.tl_map_id].unsqueeze(0)

        # Prepare LSTM input
        lstm_input = torch.stack(historical_obs, dim=1)
        lstm_input = lstm_input.permute(1, 0, 2) 
        print('lstm input')
        #print(lstm_input)

        # Get outputs from LSTM and GAT
        lstm_output = self.lstm_network(lstm_input)
        print('lstm output')
        #print(lstm_output)

        combined_output = torch.cat((lstm_output, gat_agent_feature), dim=-1)

        return combined_output

    def select_action(self, obs, gat_features):
        # Epsilon-greedy policy
        #if random.random() < self.epsilon():
        if 1 <0:
            # Exploration: choose a random action
            return random.randint(0, 3), None  # Assuming action space is Discrete(4)
        else:
            # Exploitation: use the actor network to choose an action
            combined_ouput = self.temporal_graph_forward(obs, gat_features)
            print('combined output')
            #print(combined_ouput)

            # Get action distribution from the actor network
            dist, log_probs = self.actor_network(combined_ouput)
            action = dist.sample()

            return action.item(), log_probs
        
    def learn(self, gat_features, new_gat_features, log_prob_action, obs, new_obs, reward, done):
        print('obs')
        print(obs)
        print('new obs')
        print(new_obs)
        print('forward obs')
        forward_obs = self.temporal_graph_forward(obs, gat_features)
        print(forward_obs)
        self.load_hist_buffer(new_obs)
        forward_new_obs = self.temporal_graph_forward(new_obs, new_gat_features)
        print('forward new obs')
        print(forward_new_obs)

        mask = 1 - done
        pred_value = self.critic_network(forward_obs)
        print('predicted value')
        print(pred_value)
        targ_value = reward + self.gamma * self.critic_network(forward_new_obs) * mask
        print('target value')
        print(targ_value)
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())

        # update value
        self.critic_network.optimizer.zero_grad()
        self.lstm_network.optimizer.zero_grad()
        value_loss.backward()
        self.critic_network.optimizer.step()
        self.lstm_network.optimizer.step()

        advantage = (targ_value - pred_value).detach()
        print('advantage')
        print(advantage)

        policy_loss = -advantage * log_prob_action
        policy_loss += self.entropy_weight * -log_prob_action

        self.actor_network.optimizer.zero_grad()
        policy_loss.backward()
        self.actor_network.optimizer.step()
        self.lstm_network.optimizer.step()
        
        
