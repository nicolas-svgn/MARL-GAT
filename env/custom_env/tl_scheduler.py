class TlScheduler:
    """A scheduler for traffic light events.

    This class manages a buffer of traffic light events, allowing events to be pushed into the buffer
    and popped out in a cyclic manner.
    """

    def __init__(self, tp_min, tl_id):
        """Initializes the TlScheduler.

        Args:
            tp_min (int): Minimum time period, used to determine the buffer size.
            tl_id (str): Traffic light ID, used to initialize the buffer with an event.
        """
        self.idx = 0  # Index to keep track of the current position in the buffer
        self.size = tp_min + 3  # Size of the buffer, determined by tp_min
        self.buffer = [[] for _ in range(self.size)]  # Initialize the buffer with empty lists

        # Push the initial traffic light event into the buffer
        self.push(0, (tl_id, None))

    def push(self, t_evt, tl_evt):
        """Pushes a traffic light event into the buffer.

        Args:
            t_evt (int): Time event, used to determine the position in the buffer.
            tl_evt (tuple): Traffic light event to be pushed into the buffer.
        """
        # Append the event to the buffer at the position determined by t_evt modulo size
        self.buffer[(t_evt) % self.size].append(tl_evt)

    def pop(self, idx):
        """Pops a traffic light event from the buffer.

        Args:
            idx (int): Index from which to pop the event.

        Returns:
            tl_evt (tuple or None): The popped traffic light event, or None if the buffer is empty.
        """
        try:
            # Pop the first event from the buffer at the specified index
            tl_evt = self.buffer[idx].pop(0)
        except IndexError:
            # If the buffer is empty, return None
            tl_evt = None
            # Optionally, update the index to the next position in a cyclic manner
            # self.idx = (self.idx + 1) % self.size

        return tl_evt
