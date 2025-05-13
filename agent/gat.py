import torch
import numpy as np
from .network import GATNetwork

class GATBlock:
    """
    GATBlock class for processing embedded observations using Graph Attention Networks.
    Takes embedded observations and processes them through a GATNetwork to capture
    spatial relationships between agents.
    """

    def __init__(self, device=None, learning_rate=1e-4, num_features=8, num_heads=6, output_dim=2):
        """
        Initialize the GATBlock with a GATNetwork.
        
        Args:
            device (torch.device, optional): Device to use for computation (CPU/GPU).
            learning_rate (float, optional): Learning rate for the GAT network.
            num_features (int, optional): Number of input features per node (embedding dimension).
            num_heads (int, optional): Number of attention heads in the GAT layer.
            output_dim (int, optional): Output dimension for each node.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat_network = GATNetwork(
            device=self.device,
            learning_rate=learning_rate,
            num_features=num_features,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_features = num_features
        self.output_dim = output_dim
        
    def process(self, embeddings, edge_index):
        """
        Process embedded observations through the GAT network.
        
        Args:
            embeddings (numpy.ndarray or torch.Tensor): Embedded observations with shape (num_agents, features).
            edge_index (numpy.ndarray or torch.Tensor): Edge indices with shape (2, num_edges).
            
        Returns:
            torch.Tensor: Processed features with shape (num_agents, output_dim).
        """
        # Ensure embeddings have the correct shape
        if isinstance(embeddings, np.ndarray):
            if embeddings.shape[1] != self.num_features:
                raise ValueError(f"Expected embeddings with {self.num_features} features, got {embeddings.shape[1]}")
            # Convert numpy array to torch tensor
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        elif isinstance(embeddings, torch.Tensor):
            if embeddings.shape[1] != self.num_features:
                raise ValueError(f"Expected embeddings with {self.num_features} features, got {embeddings.shape[1]}")
            embeddings = embeddings.to(self.device)
        else:
            raise TypeError("Embeddings must be a numpy array or torch tensor")
            
        # Ensure edge_index has the correct shape
        if isinstance(edge_index, np.ndarray):
            if edge_index.shape[0] != 2:
                raise ValueError(f"Expected edge_index with shape (2, num_edges), got {edge_index.shape}")
            # Convert numpy array to torch tensor
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        elif isinstance(edge_index, torch.Tensor):
            if edge_index.shape[0] != 2:
                raise ValueError(f"Expected edge_index with shape (2, num_edges), got {edge_index.shape}")
            edge_index = edge_index.to(self.device)
        else:
            raise TypeError("Edge index must be a numpy array or torch tensor")
        
        # Set model to evaluation mode
        self.gat_network.eval()
        
        # Process the embeddings through the GAT network
        with torch.no_grad():
            features = self.gat_network(embeddings, edge_index)
            
        return features
    
    def save(self, save_path, step=0, episode_count=0, rew_mean=0, len_mean=0):
        """
        Save the GAT model to a file.
        
        Args:
            save_path (str): Path to save the model.
            step (int, optional): Current training step.
            episode_count (int, optional): Number of episodes completed.
            rew_mean (float, optional): Mean reward.
            len_mean (float, optional): Mean episode length.
        """
        self.gat_network.save(save_path, step, episode_count, rew_mean, len_mean)
        
    def load(self, load_path):
        """
        Load the GAT model from a file.
        
        Args:
            load_path (str): Path to load the model from.
            
        Returns:
            tuple: Training statistics (step, episode_count, rew_mean, len_mean).
        """
        return self.gat_network.load(load_path)
    
    def to(self, device):
        """
        Move the GAT block to the specified device.
        
        Args:
            device (torch.device or str): Device to move the model to.
            
        Returns:
            GATBlock: Self for chaining.
        """
        self.device = device
        self.gat_network.to(device)
        return self
