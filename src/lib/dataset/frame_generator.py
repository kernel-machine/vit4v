import argparse
import glob
import os
import cv2
from lib.dataset.frame_buffer import FrameBuffer
import struct
import random
from abc import ABC, abstractmethod
from typing import Generator


class VideoExtractor(ABC):
    @abstractmethod
    def get_frames(self) -> Generator[cv2.Mat, None, None]:
        pass
    
class VideoFileExtractor(VideoExtractor):
    def __init__(self, video_path, window_size:int = 32):
        self.video = video_path
        self.window_size = window_size
    
    def get_frames(self) -> Generator[cv2.Mat,None,None]:
        while True:
            video_paths = glob.glob(os.path.join(self.video,"*","*.mkv"))[5:]
            #random.shuffle(video_paths)
            for video in video_paths:
                class_name = os.path.split(os.path.dirname(video))[1]
                print(f"Processing {class_name} video")

                video = cv2.VideoCapture(video)
                success = True
                while success:
                    success, frame = video.read()
                    # Scale frame to 4k keeping aspect ratio
                    if success:
                        height , width = frame.shape[:2]
                        frame = cv2.resize(frame, (int(width*0.25), int(height*0.25)))
                        yield frame

class CameraStreamExtractor(VideoExtractor):
    def __init__(self, camera_id:int):
        self.camera_id = camera_id
    
    def get_frames(self) -> Generator[cv2.Mat,None,None]:
        video = cv2.VideoCapture(self.camera_id)
        success = True
        while success:
            success, frame = video.read()
            # Scale frame to 4k keeping aspect ratio
            if success:
                yield frame

