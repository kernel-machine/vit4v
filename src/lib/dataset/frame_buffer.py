from collections.abc import Generator

class FrameBuffer:
    def __init__(self, window_size:int) -> None:
        self.buffer = []
        self.window_size = window_size

    def append(self, item: any) -> None:
        self.buffer.append(item)

    def __len__(self)->int:
        return len(self.buffer)

    def get_segments(self) -> Generator[list]:
        for i in range(len(self.buffer)//self.window_size):
            yield self.buffer[i*self.window_size:(i+1)*self.window_size]
        
        # Check if there are remaining elements
        if len(self.buffer) % self.window_size>0:
            # Loop the remaining frames
            buffer = []
            start_index = (len(self.buffer)//self.window_size)*self.window_size
            taker_index = start_index
            taker_step = 1
            while len(buffer)<self.window_size:
                buffer.append(self.buffer[taker_index])
                taker_index += taker_step
                if taker_index < start_index or taker_index >= len(self.buffer):
                    taker_step = -taker_step
                    taker_index += taker_step
            yield buffer

    def clear(self) -> None:
        self.buffer.clear()

if __name__ == "__main__":
    fb = FrameBuffer(window_size=4)
    for i in range(0,10):
        fb.append(i)

    items = fb.get_segments()
    assert next(items) == [0,1,2,3]
    assert next(items) == [4,5,6,7]
    assert next(items) == [8,9,9,8]

    fb.clear()
    for i in range(0,9):
        fb.append(i)

    items = fb.get_segments()
    assert next(items) == [0,1,2,3]
    assert next(items) == [4,5,6,7]
    assert next(items) == [8,8,8,8]

    fb.clear()
    for i in range(0,8):
        fb.append(i)

    items = fb.get_segments()
    assert next(items) == [0,1,2,3]
    assert next(items) == [4,5,6,7]






