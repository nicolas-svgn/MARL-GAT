import os
import torch
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from env.pz_env import CustomPZEnv
from env.env_graph import GraphConverter
from agent.embedder import Embedder
from agent.gat import GATBlock
from agent.agent import TGATA2CAgent

class Train:
    """
    Class to manage the training of MARL-GAT agents for traffic control.
    Handles environment setup, model initialization, and training loop.
    """
    
    def __init__(
        self,
        env_config={
            "render_mode": None,
            "gui": False,
            "log": True,
            "rnd": (False, False)
        },
        graph_path="./env/custom_env/data/9tls_3x3x3x3/plain.edg.csv",
        model_save_dir="./models",
        log_dir="./logs",
        tensorboard_dir="./runs",
        device=None,
        num_episodes=1000,
        max_steps_per_episode=1000,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_weight=0.01,
        lstm_hist_size=9,
        input_shape=(3, 12, 20),
        num_actions=4,
        eval_interval=10,
        save_interval=50,
        log_interval=1
    ):
        """
        Initialize the training setup.
        
        Args:
            env_config (dict): Configuration for the environment
            graph_path (str): Path to the graph edge file
            model_save_dir (str): Directory to save models
            log_dir (str): Directory to save logs
            tensorboard_dir (str): Directory for tensorboard logs
            device (torch.device): Device to use for training
            num_episodes (int): Number of episodes to train for
            max_steps_per_episode (int): Maximum steps per episode
            learning_rate (float): Learning rate for the models
            gamma (float): Discount factor for future rewards
            entropy_weight (float): Weight for entropy regularization
            lstm_hist_size (int): Size of historical buffer for LSTM
            input_shape (tuple): Shape of observation input (channels, height, width)
            num_actions (int): Number of possible actions
            eval_interval (int): Episodes between evaluations
            save_interval (int): Episodes between model saves
            log_interval (int): Episodes between logging to tensorboard
        """
        # Setup device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training parameters
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.lstm_hist_size = lstm_hist_size
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # Directories
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.tensorboard_dir = tensorboard_dir
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Setup tensorboard writer
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, run_id))
        
        # Setup environment
        self.env = CustomPZEnv(**env_config)
        
        # Setup graph
        self.converter = GraphConverter(graph_path)
        self.graph = self.converter.create_graph()
        self.edge_index = self.converter.get_edge_matrix(self.graph)
        self.edge_index = self.edge_index.to(self.device)
        
        # Setup shared components
        self.embedder = Embedder(device=self.device, input_shape=input_shape)
        self.gat_block = GATBlock(device=self.device)
        
        # Setup agent dict
        self.agents = {}
        self.setup_agents()
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_rewards = []
        self.avg_lengths = []
        self.agent_losses = defaultdict(list)
        self.agent_avg_rewards = defaultdict(list)
        self.global_step = 0
        
    def setup_agents(self):
        """Initialize agents for each traffic light."""
        for agent_id in self.env.agents:
            self.agents[agent_id] = TGATA2CAgent(
                tl_id=agent_id,
                device=self.device,
                lstm_hist_size=self.lstm_hist_size,
                num_actions=self.num_actions,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                entropy_weight=self.entropy_weight
            )
    
    def preprocess_observations(self, observations):
        """
        Preprocess observations through the embedder.
        
        Args:
            observations (dict): Dictionary of observations for each agent
            
        Returns:
            dict: Dictionary of embedded observations for each agent
        """
        embeddings = {}
        for agent_id, obs in observations.items():
            # Add batch dimension if needed
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            elif isinstance(obs, torch.Tensor) and obs.dim() == 3:
                obs = obs.unsqueeze(0).to(self.device)
            
            # Get embedding from embedder
            with torch.no_grad():  # No need to track gradients for embedding
                embedding = self.embedder.embed_observation(obs)
            
            embeddings[agent_id] = embedding
            
            # Store in historical buffer
            self.agents[agent_id].store_hist_buffer(embedding.detach().clone())
            
        return embeddings
    
    def process_gat(self, embeddings):
        """
        Process embeddings through GAT network.
        
        Args:
            embeddings (dict): Dictionary of embeddings for each agent
            
        Returns:
            dict: Dictionary of GAT outputs for each agent
        """
        # Arrange embeddings in order for GAT processing
        agent_ids = list(embeddings.keys())
        embedding_list = [embeddings[agent_id] for agent_id in agent_ids]
        embedding_tensor = torch.stack(embedding_list)
        
        # Process through GAT
        with torch.no_grad():  # No need to track gradients for GAT
            gat_output = self.gat_block.process(embedding_tensor, self.edge_index)
        
        # Convert back to dictionary
        gat_outputs = {}
        for i, agent_id in enumerate(agent_ids):
            gat_outputs[agent_id] = gat_output[i].detach().clone()
            
        return gat_outputs
    
    def train_episode(self):
        """
        Train for a single episode.
        
        Returns:
            tuple: (episode_reward, episode_length, agent_metrics) - Episode metrics
        """
        # Reset environment
        observations, _ = self.env.reset()
        
        # Process initial observations
        embeddings = self.preprocess_observations(observations)
        gat_outputs = self.process_gat(embeddings)
        
        # Initialize episode metrics
        total_rewards = defaultdict(float)
        episode_length = 0
        terminated = {agent_id: False for agent_id in self.env.agents}
        truncated = {agent_id: False for agent_id in self.env.agents}
        
        # Reset agent episode data
        for agent in self.agents.values():
            agent.reset_episode()
        
        # Episode loop
        while not all(terminated.values()) and not all(truncated.values()) and episode_length < self.max_steps_per_episode:
            # Select actions for each agent
            actions = {}
            action_info = {}
            
            for agent_id, agent in self.agents.items():
                if terminated[agent_id] or truncated[agent_id]:
                    continue
                
                # Prepare LSTM input
                lstm_input = agent.prepare_lstm_input(embeddings[agent_id])
                
                # Select action
                action, log_prob, entropy, value = agent.select_action(
                    lstm_input, 
                    gat_outputs[agent_id].unsqueeze(0)
                )
                
                actions[agent_id] = action
                action_info[agent_id] = (log_prob, entropy, value)
            
            # Step environment
            next_observations, rewards, new_terminated, new_truncated, infos = self.env.step(actions)
            
            # Update global step
            self.global_step += 1
            
            # Update terminated/truncated status
            terminated.update(new_terminated)
            truncated.update(new_truncated)
            
            # Process new observations
            next_embeddings = self.preprocess_observations(next_observations)
            next_gat_outputs = self.process_gat(next_embeddings)
            
            # Store experience for each agent
            for agent_id, agent in self.agents.items():
                if agent_id in actions:  # Only update agents that took actions
                    log_prob, entropy, value = action_info[agent_id]
                    reward = rewards.get(agent_id, 0)
                    total_rewards[agent_id] += reward
                    
                    agent.add_experience(reward, value, log_prob, entropy)
            
            # Update for next step
            embeddings = next_embeddings
            gat_outputs = next_gat_outputs
            episode_length += 1
        
        # Compute next state values for bootstrapping (if not terminated)
        next_values = {}
        for agent_id, agent in self.agents.items():
            if not (terminated[agent_id] or truncated[agent_id]):
                # Prepare LSTM input for final state
                lstm_input = agent.prepare_lstm_input(embeddings[agent_id])
                
                # Get value of final state
                with torch.no_grad():
                    _, state_value = agent.forward_pass(
                        lstm_input, 
                        gat_outputs[agent_id].unsqueeze(0)
                    )
                next_values[agent_id] = state_value.item()
            else:
                next_values[agent_id] = 0  # Terminal state has value 0
        
        # Update agents with complete episodes and track losses
        agent_losses = {}
        for agent_id, agent in self.agents.items():
            loss = agent.learn(
                agent.episode_rewards,
                agent.episode_values,
                agent.episode_log_probs,
                agent.episode_entropies,
                next_values.get(agent_id, 0)
            )
            agent_losses[agent_id] = loss
            self.agent_losses[agent_id].append(loss)
        
        # Collect individual agent rewards
        agent_rewards = {agent_id: sum(self.agents[agent_id].episode_rewards) for agent_id in self.agents}
        for agent_id, reward in agent_rewards.items():
            self.agent_avg_rewards[agent_id].append(reward)
            
        # Calculate episode total reward (average across agents)
        if len(total_rewards) > 0:
            episode_reward = sum(total_rewards.values()) / len(total_rewards)
        else:
            episode_reward = 0
            
        return episode_reward, episode_length, {
            'agent_losses': agent_losses,
            'agent_rewards': agent_rewards
        }
    
    def evaluate(self, num_episodes=5):
        """
        Evaluate current policy without training.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            tuple: (avg_reward, avg_length, agent_rewards) - Evaluation metrics
        """
        total_rewards = []
        total_lengths = []
        agent_rewards = defaultdict(list)
        
        for _ in range(num_episodes):
            # Reset environment
            observations, _ = self.env.reset()
            
            # Process initial observations
            embeddings = self.preprocess_observations(observations)
            gat_outputs = self.process_gat(embeddings)
            
            # Initialize episode metrics
            episode_rewards = defaultdict(float)
            episode_length = 0
            terminated = {agent_id: False for agent_id in self.env.agents}
            truncated = {agent_id: False for agent_id in self.env.agents}
            
            # Episode loop
            while not all(terminated.values()) and not all(truncated.values()) and episode_length < self.max_steps_per_episode:
                # Select actions for each agent (deterministic policy)
                actions = {}
                
                for agent_id, agent in self.agents.items():
                    if terminated[agent_id] or truncated[agent_id]:
                        continue
                    
                    # Prepare LSTM input
                    lstm_input = agent.prepare_lstm_input(embeddings[agent_id])
                    
                    # Select action (deterministic)
                    action, _, _, _ = agent.select_action(
                        lstm_input, 
                        gat_outputs[agent_id].unsqueeze(0),
                        training=False  # Use deterministic action selection
                    )
                    
                    actions[agent_id] = action
                
                # Step environment
                next_observations, rewards, new_terminated, new_truncated, _ = self.env.step(actions)
                
                # Update terminated/truncated status
                terminated.update(new_terminated)
                truncated.update(new_truncated)
                
                # Process new observations
                next_embeddings = self.preprocess_observations(next_observations)
                next_gat_outputs = self.process_gat(next_embeddings)
                
                # Update rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                # Update for next step
                embeddings = next_embeddings
                gat_outputs = next_gat_outputs
                episode_length += 1
            
            # Store agent rewards
            for agent_id, reward in episode_rewards.items():
                agent_rewards[agent_id].append(reward)
            
            # Calculate episode average reward
            if len(episode_rewards) > 0:
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
            else:
                avg_reward = 0
                
            total_rewards.append(avg_reward)
            total_lengths.append(episode_length)
        
        # Calculate average rewards per agent
        avg_agent_rewards = {agent_id: np.mean(rewards) for agent_id, rewards in agent_rewards.items()}
        
        return np.mean(total_rewards), np.mean(total_lengths), avg_agent_rewards
    
    def save_models(self, episode):
        """
        Save all models.
        
        Args:
            episode (int): Current episode number
        """
        # Save embedder and GAT models
        embedder_path = os.path.join(self.model_save_dir, f"embedder_{episode}.msgpack")
        gat_path = os.path.join(self.model_save_dir, f"gat_{episode}.msgpack")
        
        self.embedder.save(embedder_path, episode)
        self.gat_block.save(gat_path, episode)
        
        # Save agent models
        agent_dir = os.path.join(self.model_save_dir, f"agents_{episode}")
        os.makedirs(agent_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save(
                agent_dir,
                episode,
                len(self.episode_rewards),
                np.mean(self.avg_rewards[-10:]) if self.avg_rewards else 0,
                np.mean(self.avg_lengths[-10:]) if self.avg_lengths else 0
            )
    
    def load_models(self, episode):
        """
        Load all models.
        
        Args:
            episode (int): Episode number to load from
        """
        # Load embedder and GAT models
        embedder_path = os.path.join(self.model_save_dir, f"embedder_{episode}.msgpack")
        gat_path = os.path.join(self.model_save_dir, f"gat_{episode}.msgpack")
        
        if os.path.exists(embedder_path) and os.path.exists(gat_path):
            self.embedder.load(embedder_path)
            self.gat_block.load(gat_path)
        else:
            print(f"Could not find embedder or GAT models for episode {episode}")
        
        # Load agent models
        agent_dir = os.path.join(self.model_save_dir, f"agents_{episode}")
        
        if os.path.exists(agent_dir):
            for agent_id, agent in self.agents.items():
                try:
                    agent.load(agent_dir)
                except FileNotFoundError:
                    print(f"Could not find model for agent {agent_id} in episode {episode}")
        else:
            print(f"Could not find agent directory for episode {episode}")
    
    def log_to_tensorboard(self, episode, episode_reward, episode_length, agent_metrics, eval_metrics=None):
        """
        Log metrics to tensorboard.
        
        Args:
            episode (int): Current episode number
            episode_reward (float): Total reward for the episode
            episode_length (int): Length of the episode
            agent_metrics (dict): Metrics for individual agents
            eval_metrics (tuple, optional): Evaluation metrics (reward, length, agent_rewards)
        """
        # Log global metrics
        self.writer.add_scalar('train/episode_reward', episode_reward, episode)
        self.writer.add_scalar('train/episode_length', episode_length, episode)
        
        # Calculate and log running averages
        window_size = min(10, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window_size:])
        avg_length = np.mean(self.episode_lengths[-window_size:])
        self.writer.add_scalar('train/avg_reward_10', avg_reward, episode)
        self.writer.add_scalar('train/avg_length_10', avg_length, episode)
        
        # Log agent-specific metrics
        for agent_id, loss in agent_metrics['agent_losses'].items():
            self.writer.add_scalar(f'train/agent_{agent_id}/loss', loss, episode)
            self.writer.add_scalar(f'train/agent_{agent_id}/episode_reward', 
                                   agent_metrics['agent_rewards'][agent_id], episode)
        
        # Log evaluation metrics if available
        if eval_metrics:
            eval_reward, eval_length, eval_agent_rewards = eval_metrics
            self.writer.add_scalar('eval/avg_reward', eval_reward, episode)
            self.writer.add_scalar('eval/avg_length', eval_length, episode)
            
            for agent_id, reward in eval_agent_rewards.items():
                self.writer.add_scalar(f'eval/agent_{agent_id}/reward', reward, episode)
        
        # Flush writer to ensure logs are saved
        self.writer.flush()
    
    def plot_results(self):
        """Plot training results."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.avg_rewards)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(2, 2, 2)
        plt.plot(self.avg_lengths)
        plt.title('Average Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        plt.subplot(2, 2, 3)
        for agent_id, rewards in self.agent_avg_rewards.items():
            if len(rewards) > 0:
                window_size = min(10, len(rewards))
                smoothed_rewards = [np.mean(rewards[max(0, i-window_size):i+1]) 
                                    for i in range(len(rewards))]
                plt.plot(smoothed_rewards, label=f'Agent {agent_id}')
        plt.title('Agent Rewards (Smoothed)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for agent_id, losses in self.agent_losses.items():
            if len(losses) > 0:
                window_size = min(10, len(losses))
                smoothed_losses = [np.mean(losses[max(0, i-window_size):i+1]) 
                                  for i in range(len(losses))]
                plt.plot(smoothed_losses, label=f'Agent {agent_id}')
        plt.title('Agent Losses (Smoothed)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_results.png'))
        plt.close()
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        start_time = time.time()
        
        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()
            
            # Train for one episode
            episode_reward, episode_length, agent_metrics = self.train_episode()
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Calculate running averages
            window_size = min(10, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-window_size:])
            avg_length = np.mean(self.episode_lengths[-window_size:])
            self.avg_rewards.append(avg_reward)
            self.avg_lengths.append(avg_length)
            
            # Print progress
            episode_time = time.time() - episode_start_time
            print(f"Episode {episode}/{self.num_episodes} - " 
                  f"Reward: {episode_reward:.2f} - "
                  f"Length: {episode_length} - "
                  f"Avg Reward (10): {avg_reward:.2f} - "
                  f"Time: {episode_time:.2f}s")
            
            # Log to tensorboard
            if episode % self.log_interval == 0:
                eval_metrics = None
                if episode % self.eval_interval == 0:
                    eval_reward, eval_length, eval_agent_rewards = self.evaluate()
                    eval_metrics = (eval_reward, eval_length, eval_agent_rewards)
                    print(f"Evaluation - Avg Reward: {eval_reward:.2f} - Avg Length: {eval_length:.2f}")
                
                self.log_to_tensorboard(episode, episode_reward, episode_length, agent_metrics, eval_metrics)
            
            # Save models periodically
            if episode % self.save_interval == 0:
                self.save_models(episode)
                self.plot_results()
        
        # Final save and plot
        self.save_models(self.num_episodes)
        self.plot_results()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.avg_rewards, self.avg_lengths


if __name__ == "__main__":
    # Set PyTorch to handle gradient computation carefully
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create trainer
    trainer = Train(
        env_config={
            "render_mode": None,
            "gui": False,
            "log": True,
            "rnd": (False, False)
        },
        num_episodes=500,
        max_steps_per_episode=1000,
        eval_interval=20,
        save_interval=50,
        log_interval=1  # Log every episode
    )
    
    # Start training
    trainer.train()
