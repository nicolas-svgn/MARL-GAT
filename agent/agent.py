import torch
import torch.nn.functional as F
import numpy as np
import os
from .network import LSTMNetwork, ActorNetwork, CriticNetwork
from .hist_buffer import HistoricalBuffer

class TGATA2CAgent:
    """
    Traffic Graph Attention A2C Agent.
    
    This agent combines LSTM for temporal processing with GAT outputs for spatial processing
    of traffic network data. It implements the Advantage Actor-Critic (A2C) algorithm
    for learning optimal traffic control policies.
    """
    
    def __init__(self, 
                 tl_id, 
                 device=None,
                 lstm_input_dim=8,
                 lstm_hidden_dim=50,
                 lstm_output_dim=2,
                 lstm_num_layers=4,
                 lstm_hist_size=9,
                 actor_input_dim=4,
                 actor_hidden_dim=64,
                 num_actions=4,
                 critic_input_dim=4,
                 critic_hidden_dim=64,
                 learning_rate=1e-4,
                 gamma=0.99,
                 entropy_weight=0.01):
        """
        Initialize the TGATA2C Agent.
        
        Args:
            tl_id (str): Traffic light ID this agent controls
            device (torch.device, optional): Device to use for computation
            lstm_input_dim (int): Dimension of LSTM input (embeddings)
            lstm_hidden_dim (int): Dimension of LSTM hidden state
            lstm_output_dim (int): Dimension of LSTM output
            lstm_num_layers (int): Number of LSTM layers
            lstm_hist_size (int): Size of historical buffer for LSTM
            actor_input_dim (int): Dimension of actor network input (LSTM output + GAT output)
            actor_hidden_dim (int): Dimension of actor network hidden layer
            num_actions (int): Number of possible actions (traffic light phases)
            critic_input_dim (int): Dimension of critic network input (LSTM output + GAT output)
            critic_hidden_dim (int): Dimension of critic network hidden layer
            learning_rate (float): Learning rate for all networks
            gamma (float): Discount factor for future rewards
            entropy_weight (float): Weight of entropy term in loss function
        """
        self.tl_id = tl_id
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.num_actions = num_actions
        
        # Initialize networks
        self.lstm_network = LSTMNetwork(
            device=self.device,
            learning_rate=learning_rate,
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            historical_buffer_size=lstm_hist_size,
            output_dim=lstm_output_dim
        )
        
        self.actor_network = ActorNetwork(
            device=self.device,
            learning_rate=learning_rate,
            input_dim=actor_input_dim,
            hidden_dim=actor_hidden_dim,
            num_actions=num_actions
        )
        
        self.critic_network = CriticNetwork(
            device=self.device,
            learning_rate=learning_rate,
            input_dim=critic_input_dim,
            hidden_dim=critic_hidden_dim
        )
        
        # Initialize historical buffer
        self.hist_buffer = HistoricalBuffer(tl_id, size=lstm_hist_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_entropies = []
    
    def forward_pass(self, lstm_input, gat_output):
        """
        Perform a forward pass through the agent's networks.
        
        Args:
            lstm_input (torch.Tensor): Input to the LSTM network [batch_size, seq_len, input_dim]
            gat_output (torch.Tensor): Output from the GAT network [batch_size, gat_output_dim]
            
        Returns:
            tuple: (action_dist, state_value) - Action distribution and state value
        """
        # Pass input through LSTM
        lstm_output = self.lstm_network(lstm_input)
        
        # Combine LSTM output with GAT output
        combined_features = torch.cat([lstm_output, gat_output], dim=-1)
        
        # Get action distribution from actor network
        action_dist = self.actor_network(combined_features)
        
        # Get value prediction from critic network
        state_value = self.critic_network(combined_features)
        
        return action_dist, state_value
    
    def select_action(self, lstm_input, gat_output, training=True):
        """
        Select an action based on current state.
        
        Args:
            lstm_input (torch.Tensor): Input to the LSTM network
            gat_output (torch.Tensor): Output from the GAT network
            training (bool): Whether the agent is training (affects action selection)
            
        Returns:
            tuple: (action, action_log_prob, entropy, value) - Selected action and related information
        """
        # Set networks to evaluation mode
        self.lstm_network.eval()
        self.actor_network.eval()
        self.critic_network.eval()
        
        with torch.no_grad():
            # Forward pass to get action distribution and value
            action_dist, state_value = self.forward_pass(lstm_input, gat_output)
            
            # Sample action from distribution if training, otherwise take best action
            if training:
                action = action_dist.sample()
            else:
                # During evaluation, take the action with highest probability
                action = torch.argmax(action_dist.probs)
            
            # Get log probability and entropy of selected action
            action_log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
        
        # Convert to numpy/python types for environment interaction
        action_np = action.item()
        value_np = state_value.item()
        
        return action_np, action_log_prob, entropy, value_np
    
    def learn(self, rewards, values, log_probs, entropies, next_value=0):
        """
        Update network parameters using the A2C algorithm.
        
        Args:
            rewards (list): List of rewards received during episode
            values (list): List of state values predicted during episode
            log_probs (list): List of log probabilities of selected actions
            entropies (list): List of entropies of action distributions
            next_value (float): Value of the last state (0 if terminal)
            
        Returns:
            float: Total loss for the update
        """
        # Set networks to training mode
        self.lstm_network.train()
        self.actor_network.train()
        self.critic_network.train()
        
        # Convert lists to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards, next_value)
        advantages = returns - values
        
        # Calculate actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(values, returns)
        
        # Calculate entropy loss (to encourage exploration)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = actor_loss + critic_loss + self.entropy_weight * entropy_loss
        
        # Optimize all networks
        self.lstm_network.optimizer.zero_grad()
        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()
        
        total_loss.backward()
        
        self.lstm_network.optimizer.step()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()
        
        return total_loss.item()
    
    def _compute_returns(self, rewards, next_value):
        """
        Compute discounted returns for a sequence of rewards.
        
        Args:
            rewards (torch.Tensor): Tensor of rewards
            next_value (float): Value of the state after the last reward (0 if terminal)
            
        Returns:
            torch.Tensor: Tensor of discounted returns
        """
        returns = torch.zeros_like(rewards)
        R = next_value
        
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
            
        return returns
    
    def store_hist_buffer(self, embedding):
        """
        Store an embedding in the historical buffer.
        
        Args:
            embedding (torch.Tensor): Embedding to store
        """
        self.hist_buffer.store(embedding)
    
    def get_hist_buffer(self):
        """
        Get the current historical buffer.
        
        Returns:
            list: List of embeddings in the buffer
        """
        return self.hist_buffer.get()
    
    def prepare_lstm_input(self, current_embedding):
        """
        Prepare input for LSTM by combining historical buffer with current embedding.
        
        Args:
            current_embedding (torch.Tensor): Current observation embedding
            
        Returns:
            torch.Tensor: Tensor of shape [1, seq_len, embedding_dim] for LSTM input
        """
        # Get historical buffer
        buffer = self.get_hist_buffer()
        
        # If buffer is not full, pad with zeros
        if len(buffer) < self.hist_buffer.max_size:
            # Create padding tensor
            pad_size = self.hist_buffer.max_size - len(buffer)
            padding = [torch.zeros_like(current_embedding) for _ in range(pad_size)]
            
            # Combine padding, buffer, and current embedding
            seq = padding + buffer + [current_embedding]
        else:
            # Combine buffer and current embedding
            seq = buffer + [current_embedding]
        
        # Convert sequence to tensor
        seq_tensor = torch.stack(seq).unsqueeze(0)  # Add batch dimension
        
        return seq_tensor
    
    def reset_episode(self):
        """Reset episode-specific data."""
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_entropies = []
    
    def add_experience(self, reward, value, log_prob, entropy):
        """
        Add experience from a step to the episode history.
        
        Args:
            reward (float): Reward received
            value (float): Value predicted
            log_prob (torch.Tensor): Log probability of action
            entropy (torch.Tensor): Entropy of action distribution
        """
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
        self.episode_log_probs.append(log_prob)
        self.episode_entropies.append(entropy)
    
    def save(self, save_dir, step=0, episode_count=0, rew_mean=0, len_mean=0):
        """
        Save all network models.
        
        Args:
            save_dir (str): Directory to save models in
            step (int): Current training step
            episode_count (int): Number of episodes completed
            rew_mean (float): Mean reward
            len_mean (float): Mean episode length
        """
        os.makedirs(save_dir, exist_ok=True)
        
        lstm_path = os.path.join(save_dir, f"{self.tl_id}_lstm.msgpack")
        actor_path = os.path.join(save_dir, f"{self.tl_id}_actor.msgpack")
        critic_path = os.path.join(save_dir, f"{self.tl_id}_critic.msgpack")
        
        self.lstm_network.save(lstm_path, step, episode_count, rew_mean, len_mean)
        self.actor_network.save(actor_path, step, episode_count, rew_mean, len_mean)
        self.critic_network.save(critic_path, step, episode_count, rew_mean, len_mean)
    
    def load(self, save_dir):
        """
        Load all network models.
        
        Args:
            save_dir (str): Directory to load models from
            
        Returns:
            tuple: (step, episode_count, rew_mean, len_mean) from the actor network
        """
        lstm_path = os.path.join(save_dir, f"{self.tl_id}_lstm.msgpack")
        actor_path = os.path.join(save_dir, f"{self.tl_id}_actor.msgpack")
        critic_path = os.path.join(save_dir, f"{self.tl_id}_critic.msgpack")
        
        self.lstm_network.load(lstm_path)
        self.actor_network.load(actor_path)
        stats = self.critic_network.load(critic_path)
        
        return stats
    
    def to(self, device):
        """
        Move all networks to the specified device.
        
        Args:
            device (torch.device): Device to move to
            
        Returns:
            TGATA2CAgent: Self for chaining
        """
        self.device = device
        self.lstm_network.to(device)
        self.actor_network.to(device)
        self.critic_network.to(device)
        return self
