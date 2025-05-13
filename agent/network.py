import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv
import msgpack
from .utils import msgpack_numpy_patch

# Apply a patch to msgpack to handle numpy arrays
msgpack_numpy_patch()

# Mapping of traffic light IDs to indices
tls_mapping = {
    'gneJ1': 0, 'gneJ10': 1, 'gneJ13': 2, 'gneJ15': 3, 'gneJ18': 4,
    'gneJ3': 5, 'gneJ20': 6, 'gneJ5': 7, 'gneJ8': 8
}

class Network(nn.Module):
    """Base network class with save and load functionality."""

    def __init__(self, device=None):
        super(Network, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, s):
        """Forward pass of the network. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, save_path, step, episode_count, rew_mean, len_mean):
        """Saves the network parameters and training statistics to a file.

        Args:
            save_path (str): Path to save the parameters.
            step (int): Current training step.
            episode_count (int): Number of episodes completed.
            rew_mean (float): Mean reward.
            len_mean (float): Mean episode length.
        """
        params_dict = {
            'parameters': {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()},
            'step': step, 'episode_count': episode_count, 'rew_mean': rew_mean, 'len_mean': len_mean
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(msgpack.dumps(params_dict))

    def load(self, load_path):
        """Loads the network parameters and training statistics from a file.

        Args:
            load_path (str): Path to load the parameters from.

        Returns:
            tuple: Training statistics (step, episode_count, rew_mean, len_mean).
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_dict = msgpack.loads(f.read())

        parameters = {k: torch.as_tensor(v, device=self.device) for k, v in params_dict['parameters'].items()}
        self.load_state_dict(parameters)

        return params_dict['step'], params_dict['episode_count'], params_dict['rew_mean'], params_dict['len_mean']


class BaseNetwork(Network):
    """A base convolutional neural network for processing DTSE observations."""

    def __init__(self, device=None, learning_rate=1e-4, input_shape=(3, 12, 20)):
        super(BaseNetwork, self).__init__(device)

        self.input_shape = input_shape
        self.learning_rate = learning_rate

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
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
        
        # Calculate the size of the flattened feature vector after the convolutional layers
        # Formula: floor((input_size - kernel_size + 2*padding) / stride) + 1 with padding = 0
        conv1_height = ((input_shape[1] - 4 + 2*0) // 2) + 1  # ((12 - 4) // 2) + 1 = (8 // 2) + 1 = 4 + 1 = 5
        conv1_width  = ((input_shape[2] - 4 + 2*0) // 2) + 1  # ((20 - 4) // 2) + 1 = (16 // 2) + 1 = 8 + 1 = 9
        conv2_height = ((conv1_height - 2 + 2*0) // 1) + 1    # ((5 - 2) // 1) + 1 = (3 // 1) + 1 = 3 + 1 = 4
        conv2_width  = ((conv1_width  - 2 + 2*0) // 1) + 1    # ((9 - 2) // 1) + 1 = (7 // 1) + 1 = 7 + 1 = 8
        
        self.flatten_dim = 32 * conv2_height * conv2_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)  # Output 8-dimensional embedding

        # Weight initialization
        self._initialize_weights()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('elu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 12, 20].

        Returns:
            torch.Tensor: Output embedding of shape [batch_size, 8].
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        if x.dim() == 3:  
            x = x.unsqueeze(0)
            
        # Make sure x is on the correct device
        x = x.to(self.device)
        
        # Convolutional layers with ELU activation
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers with ELU activation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)  # No activation after the last layer
        return x


class GATNetwork(Network):
    """Graph Attention Network (GAT) for sharing information between agents."""

    def __init__(self, device=None, learning_rate=1e-4, num_features=8, num_heads=6, output_dim=2):
        super(GATNetwork, self).__init__(device)
        
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Graph Attention Layer
        self.conv1 = GATv2Conv(
            in_channels=num_features, 
            out_channels=output_dim, 
            heads=num_heads, 
            concat=False,
            dropout=0.2  
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        """Forward pass through the GAT network.

        Args:
            x (torch.Tensor): Input node features of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Edge indices for the graph of shape [2, num_edges].

        Returns:
            torch.Tensor: Output node features of shape [num_nodes, output_dim].
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            
        # Make sure inputs are on the correct device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Forward through GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        
        return x


class LSTMNetwork(Network):
    """LSTM network for temporal sequence modeling."""

    def __init__(self, device=None, learning_rate=1e-4, input_dim=8, hidden_dim=50, 
                 num_layers=4, historical_buffer_size=9, output_dim=2):
        super(LSTMNetwork, self).__init__(device)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.historical_buffer_size = historical_buffer_size
        
        # LSTM Layer - note we use batch_first=True for easier handling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Use batch_first=True for easier handling
            dropout=0.1 if num_layers > 1 else 0  # Add dropout between LSTM layers
        )
        
        # Fully Connected Layer for final output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _initialize_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """Forward pass through the LSTM network.

        Args:
            x (torch.Tensor): Input sequence tensor of shape 
                             [batch_size, sequence_length, input_dim]
                             where sequence_length should equal historical_buffer_size + 1 (current)

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        # Make sure x is on the correct device
        x = x.to(self.device)
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Verify sequence length
        expected_seq_len = self.historical_buffer_size + 1  # Current + historical
        assert seq_len == expected_seq_len, f"Expected sequence length {expected_seq_len}, got {seq_len}"
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out shape: [batch_size, seq_len, hidden_dim]
        
        # Take the output of the last time step
        out = out[:, -1, :]  # shape: [batch_size, hidden_dim]
        
        # Pass the output through the fully connected layer
        out = self.fc(out)  # shape: [batch_size, output_dim]
        
        return out


class ActorNetwork(Network):
    """Actor network for policy-based reinforcement learning."""

    def __init__(self, device=None, learning_rate=1e-4, input_dim=4, hidden_dim=64, num_actions=4):
        super(ActorNetwork, self).__init__(device)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the actor network.

        Args:
            x (torch.Tensor): Input state tensor of shape [batch_size, input_dim].

        Returns:
            Categorical: Action distribution.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        # Make sure x is on the correct device
        x = x.to(self.device)
        
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Create a categorical distribution
        dist = Categorical(probs)
        
        return dist


class CriticNetwork(Network):
    """Critic network for value-based reinforcement learning."""

    def __init__(self, device=None, learning_rate=1e-4, input_dim=4, hidden_dim=64):
        super(CriticNetwork, self).__init__(device)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single value
        
        # Initialize weights
        self._initialize_weights()
        
        # Loss Function
        self.loss = nn.SmoothL1Loss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the critic network.

        Args:
            x (torch.Tensor): Input state tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: State value of shape [batch_size, 1].
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        # Make sure x is on the correct device
        x = x.to(self.device)
        
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        
        return value
