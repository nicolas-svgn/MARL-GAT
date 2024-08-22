from collections import deque

class HistoricalBuffer:
    def __init__(self, tl_id, size=9):
        self.buffer = deque(maxlen=size)
        self.tl_id = tl_id

    def store(self, obs):
        self.buffer.append(obs)

    def get(self):
        return list(self.buffer)  # Convert deque to list
