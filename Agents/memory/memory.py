class Memory:
    def __init__(self):
        self.memory = []

    def append(self, item):
        self.memory.append(item)

    def clear(self):
        self.memory.clear()

    def __getitem__(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)
