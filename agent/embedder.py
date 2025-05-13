import torch
import numpy as np
from .network import BaseNetwork

class Embedder:
    """
    Embedder class for encoding DTSE observations into fixed-size embeddings.
    Uses a BaseNetwork to transform 3D observations (3,12,20) into 1D embeddings (8).
    """

    def __init__(self, device=None, learning_rate=1e-4, input_shape=(3, 12, 20)):
        """
        Initialize the Embedder with a BaseNetwork.
        
        Args:
            device (torch.device, optional): Device to use for computation (CPU/GPU).
            learning_rate (float, optional): Learning rate for the base network.
            input_shape (tuple, optional): Shape of input observations (channels, height, width).
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_network = BaseNetwork(
            device=self.device,
            learning_rate=learning_rate,
            input_shape=input_shape
        )
        self.input_shape = input_shape
        
    def embed_observation(self, observation):
        """
        Embed a single agent's observation into an 8-dimensional vector.
        
        Args:
            observation (numpy.ndarray or torch.Tensor): Observation with shape (3, 12, 20).
            
        Returns:
            torch.Tensor: Embedded observation with shape (8).
        """
        # Ensure observation has the correct shape
        if isinstance(observation, np.ndarray):
            if observation.shape != self.input_shape:
                raise ValueError(f"Expected observation shape {self.input_shape}, got {observation.shape}")
            # Convert numpy array to torch tensor
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        elif isinstance(observation, torch.Tensor):
            if tuple(observation.shape) != self.input_shape:
                raise ValueError(f"Expected observation shape {self.input_shape}, got {tuple(observation.shape)}")
            observation = observation.to(self.device)
        else:
            raise TypeError("Observation must be a numpy array or torch tensor")
        
        # Set model to evaluation mode
        self.base_network.eval()
        
        # Process the observation through the base network
        with torch.no_grad():
            embedding = self.base_network(observation.unsqueeze(0)).squeeze(0)
            
        return embedding
    
    def embed_observations(self, observations):
        """
        Embed multiple agents' observations into a tensor of embeddings.
        
        Args:
            observations (list, numpy.ndarray, or torch.Tensor): List or batch of observations,
                each with shape (3, 12, 20).
                
        Returns:
            torch.Tensor: Embedded observations with shape (num_agents, 8).
        """
        # Handle different input types
        if isinstance(observations, list):
            # Convert list of observations to batch tensor
            batch = []
            for obs in observations:
                if isinstance(obs, np.ndarray):
                    batch.append(torch.tensor(obs, dtype=torch.float32, device=self.device))
                elif isinstance(obs, torch.Tensor):
                    batch.append(obs.to(self.device))
                else:
                    raise TypeError("Each observation must be a numpy array or torch tensor")
            observations_tensor = torch.stack(batch)
            
        elif isinstance(observations, np.ndarray):
            if observations.ndim != 4:
                raise ValueError(f"Expected 4D array (batch, channels, height, width), got shape {observations.shape}")
            observations_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
            
        elif isinstance(observations, torch.Tensor):
            if observations.dim() != 4:
                raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {tuple(observations.shape)}")
            observations_tensor = observations.to(self.device)
            
        else:
            raise TypeError("Observations must be a list, numpy array, or torch tensor")
        
        # Set model to evaluation mode
        self.base_network.eval()
        
        # Process the batch of observations
        with torch.no_grad():
            embeddings = self.base_network(observations_tensor)
            
        return embeddings
    
    def save(self, save_path, step=0, episode_count=0, rew_mean=0, len_mean=0):
        """
        Save the embedder model to a file.
        
        Args:
            save_path (str): Path to save the model.
            step (int, optional): Current training step.
            episode_count (int, optional): Number of episodes completed.
            rew_mean (float, optional): Mean reward.
            len_mean (float, optional): Mean episode length.
        """
        self.base_network.save(save_path, step, episode_count, rew_mean, len_mean)
        
    def load(self, load_path):
        """
        Load the embedder model from a file.
        
        Args:
            load_path (str): Path to load the model from.
            
        Returns:
            tuple: Training statistics (step, episode_count, rew_mean, len_mean).
        """
        return self.base_network.load(load_path)
    
    def to(self, device):
        """
        Move the embedder to the specified device.
        
        Args:
            device (torch.device or str): Device to move the model to.
            
        Returns:
            Embedder: Self for chaining.
        """
        self.device = device
        self.base_network.to(device)
        return self
