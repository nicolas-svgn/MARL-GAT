from collections import deque

class HistoricalBuffer:
    """A buffer to store historical observations for a traffic light."""
    
    def __init__(self, tl_id, size=9):
        """Initializes the HistoricalBuffer.
        
        Args:
            tl_id (str): The ID of the traffic light associated with this buffer.
            size (int): The maximum size of the buffer. Default is 9.
        """
        self.buffer = deque(maxlen=size)  # Initialize a deque with a maximum length
        self.tl_id = tl_id  # Store the traffic light ID
        self.max_size = size  # Store the maximum size for reference
        
    def store(self, obs):
        """Stores an observation in the buffer.
        
        Args:
            obs: The observation to store. This can be any data type.
        """
        self.buffer.append(obs)  # Append the observation to the buffer
        
    def get(self):
        """Retrieves all observations from the buffer.
        
        Returns:
            list: A list of all observations currently in the buffer.
        """
        return list(self.buffer)  # Convert the deque to a list and return it
    
    def is_full(self):
        """Checks if the buffer is at maximum capacity.
        
        Returns:
            bool: True if the buffer contains max_size elements, False otherwise.
        """
        return len(self.buffer) == self.max_size
    
    def clear(self):
        """Clears all observations from the buffer."""
        self.buffer.clear()
        
    def __len__(self):
        """Returns the current number of observations in the buffer.
        
        Returns:
            int: The number of observations in the buffer.
        """
        return len(self.buffer)
