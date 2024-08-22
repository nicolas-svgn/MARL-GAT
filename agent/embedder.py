from .network2 import BaseNetwork

import os
import torch
import numpy as np

class Embedder:
    def __init__(self, device, input_dim, lr_base):
        # Networks
        self.device = device
        self.lr = lr_base
        self.input_dim = input_dim

        """self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.load = load

        self.step = 0
        self.resume_step = 0
        self.episode_count = 0

        path = 'TGATA2C' + tl_id
        self.save_path = save_dir + path + '_' + 'model.pack'
        self.summary_writer = SummaryWriter(log_dir + path + '/')

        self.start_time = time.time()"""

        self.base_network = BaseNetwork(self.device, self.input_dim, self.lr).to(device)

    def load_model(self):
        if self.load and os.path.exists(self.save_path):
            print()
            print("Resume training from " + self.save_path + "...")
            self.resume_step, self.episode_count, rew_mean, len_mean = self.base_network.load(self.save_path)
            [self.ep_info_buffer.append({'r': rew_mean, 'l': len_mean}) for _ in range(np.min([self.episode_count, self.ep_info_buffer.maxlen]))]
            print("Step: ", self.resume_step * self.n_env, ", Episodes: ", self.episode_count, ", Avg Rew: ", rew_mean, ", Avg Ep Len: ", len_mean)

            self.update_target_network(force=True)
            self.step = self.resume_step

    def save_model(self):
        if self.step % self.save_frequency == 0 and self.step > self.resume_step:
            print()
            print("Saving model...")
            self.base_network.save(self.save_path, self.step, self.episode_count, self.info_mean('r'), self.info_mean('l'))
            print("OK!")

    def graph_embed_state(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        all_agent_obs = torch.stack([self.base_network(state[i]) for i in range(9)])  # Shape: (9, 1, 8)
        all_agent_obs = all_agent_obs.view(9, 8) 

        return all_agent_obs
    
    def embed_agent_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return self.base_network(obs)