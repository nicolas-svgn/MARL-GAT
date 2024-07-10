class TlScheduler:
    def __init__(self, tp_min, tl_id):
        self.idx = 0
        self.size = tp_min + 3
        self.buffer = [[] for _ in range(self.size)]
        self.push(0, (tl_id, None))

    def push(self, t_evt, tl_evt):
        self.buffer[(t_evt) % self.size].append(tl_evt)

    def pop(self, idx):
        try:
            tl_evt = self.buffer[idx].pop(0)
        except IndexError:
            tl_evt = None
            #self.idx = (self.idx + 1) % self.size
        return tl_evt
