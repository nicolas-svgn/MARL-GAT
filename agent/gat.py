from .network2 import GATNetwork

import os
import torch
import numpy as np

class GATBlock:
    def __init__(self, device, lr_gat, edge_index):

        self.device = device
        self.lr_gat = lr_gat
        self.edge_index = edge_index

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

        self.gat_network = GATNetwork(self.lr_gat).to(self.device)

    def load_model(self):
        if self.load and os.path.exists(self.save_path):
            print()
            print("Resume training from " + self.save_path + "...")
            self.resume_step, self.episode_count, rew_mean, len_mean = self.gat_network.load(self.save_path)
            [self.ep_info_buffer.append({'r': rew_mean, 'l': len_mean}) for _ in range(np.min([self.episode_count, self.ep_info_buffer.maxlen]))]
            print("Step: ", self.resume_step, ", Episodes: ", self.episode_count, ", Avg Rew: ", rew_mean, ", Avg Ep Len: ", len_mean)

            self.step = self.resume_step

    def save_model(self):
        if self.step % self.save_frequency == 0 and self.step > self.resume_step:
            print()
            print("Saving model...")
            self.gat_network.save(self.save_path, self.step, self.episode_count, self.info_mean('r'), self.info_mean('l'))
            print("OK!")

    def gat_output(self, graph_features):
        gat_output = self.gat_network(graph_features, self.edge_index)
        return gat_output