import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_geometric.nn import GATv2Conv

import msgpack
from .utils import msgpack_numpy_patch
msgpack_numpy_patch()

tls_mapping = {'gneJ1': 0, 'gneJ10': 1, 'gneJ13': 2, 'gneJ15': 3, 'gneJ18': 4, 'gneJ3': 5, 'gneJ20': 6, 'gneJ5': 7, 'gneJ8': 8}


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, s):
        raise NotImplementedError

    def save(self, save_path, step, episode_count, rew_mean, len_mean):
        params_dict = {
            'parameters': {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()},
            'step': step, 'episode_count': episode_count, 'rew_mean': rew_mean, 'len_mean': len_mean
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(msgpack.dumps(params_dict))

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_dict = msgpack.loads(f.read())

        parameters = {k: torch.as_tensor(v, device=self.device) for k, v in params_dict['parameters'].items()}
        self.load_state_dict(parameters)

        return params_dict['step'], params_dict['episode_count'], params_dict['rew_mean'], params_dict['len_mean']


"""class BaseNetwork(Network):
    def __init__(self, device, input_dim, learning_rate):
        super(BaseNetwork, self).__init__()

        self.device = device
        self.learning_rate = learning_rate

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, kernel_size=(4, 4), stride=(2, 2)),
            nn.ELU(alpha=1.0),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ELU(alpha=1.0)
        ).to(device)

        # Dynamic Calculation of Linear Input Size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim).to(device)
            conv_output = self.conv_layers(dummy_input)
            print("Shape after conv layers:", conv_output.shape)
            flattened_size = conv_output.view(1, -1).size(1)
            print("Flattened size:", flattened_size)

        # Fully Connected Layers (with dynamic input size)
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ELU(inplace=False),
            nn.Linear(128, 64),
            nn.ELU(inplace=False),
            nn.Linear(64, 32),
            nn.ELU(inplace=False),
            nn.Linear(32, 16),
            nn.ELU(inplace=False),
            nn.Linear(16, 8),
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, s):
        x = self.conv_layers(s)
        x = x.view(x.size(0), -1)  # Flatten while preserving the batch dimension
        v = self.fc_layers(x)
        return v"""

class BaseNetwork(Network):
    def __init__(self, device, input_dim, learning_rate):
        super(BaseNetwork, self).__init__()

        self.device = device
        self.learning_rate = learning_rate

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(4, 4),
            stride=(2, 2)
        )
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(1, 1)
        )
        # Activation function
        self.elu = nn.ELU()
        
        # Calculate the size of the flattened feature vector after the convolutional layers
        # Input dimensions: (channels=3, height=12, width=20)
        # After conv1: (channels=16, height=5, width=9)
        # After conv2: (channels=32, height=4, width=8)
        self.flatten_dim = 32 * 4 * 8  # 1024
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        # Convolutional layers with ELU activation
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.elu(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ELU activation
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)  # No activation after the last layer
        return x

    
class GATNetwork(Network):
    def __init__(self, learning_rate, num_features=8, num_heads=6, output_dim=2):
        super().__init__()

        self.learning_rate = learning_rate
        
        # Input to Output Layer (directly)
        self.conv1 = GATv2Conv(num_features, output_dim, heads=num_heads, concat=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Apply ELU activation
        x = F.dropout(x, p=0.6, training=self.training)  # Apply dropout during training
        return x
    

    
class ActorNetwork(Network):
    def __init__(self, learning_rate, input_dim, hidden_dim=64, num_actions=4):
        super(ActorNetwork, self).__init__()

        self.num_actions = num_actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

        self.learning_rate = learning_rate

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
        dist = Categorical(probs)          # Create a categorical distribution
        return dist

    
class CriticNetwork(Network):
    def __init__(self, learning_rate, input_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()

        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single value representing the state value

        # Loss Function
        self.loss = nn.SmoothL1Loss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value
    
class LSTMNetwork(nn.Module):
    def __init__(self, learning_rate, input_dim=8, num_layers=4, historical_buffer_size=9, hidden_dim=50, output_dim=2):
        super(LSTMNetwork, self).__init__()

        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.historical_buffer_size = historical_buffer_size 

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=False)

        # Fully Connected Layer for final output
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, input_dim)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (sequence_length, batch_size, hidden_dim)

        # Take the output of the last time step
        out = out[-1, :, :]  # shape: (batch_size, hidden_dim)

        # Pass the output through the fully connected layer
        out = self.fc(out)  # shape: (batch_size, output_dim)

        return out
