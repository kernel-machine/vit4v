from collections.abc import Generator

class FifoFrames:
    def __init__(self, max_size: int = -1) -> None:
        self.max_size = max_size
        self.buffer = []

    def append(self, item: any) -> None:
        if self.max_size > 0 and len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def __len__(self)->int:
        return len(self.buffer)
        
    def reverse_loop_generator(self) -> Generator[any]:
        if len(self.buffer) == 0:
            raise StopIteration
        while True:
            for i in range(len(self.buffer)-1, -1, -1):
                yield self.buffer[i]
            for i in range(len(self.buffer)):
                yield self.buffer[i]

