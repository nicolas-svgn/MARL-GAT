from collections import deque

class HistoricalBuffer:
    def __init__(self, tl_id, size=9):
        self.buffer = deque(maxlen=size)
        self.tl_id = tl_id

    def store(self, obs):
        self.buffer.append(obs.clone())

    def get(self):
        return [obs.clone() for obs in self.buffer]  # Convert deque to list
