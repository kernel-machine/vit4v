from lib.dataset.VideoSegmenter2 import VideoSegmenterGen
import argparse
import glob
import os
import random
import cv2
from pathlib import Path
from lib.dataset.fifo_frames import FifoFrames
import shutil
from multiprocessing.pool import ThreadPool

'''
From RAW videos this code extract the segments
'''
parser = argparse.ArgumentParser(description="Extract segments from videos")
parser.add_argument(
    "--video_path",
    type=str,
    help="Folder containing the folders varroa infested and varroa_free with original videos",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Folder where the output videos are stored",
    required=True,
)
parser.add_argument(
    "--validation_split", type=float, default=0.3, help="Validation split"
)
parser.add_argument(
    "--window_size",
    type=int,
    default=16,
    help="Number of frames that are processed by the model",
)
parser.add_argument("--seed", type=int, default=1234, help="Seed for random actions")
parser.add_argument("--show_preview", default=False, action="store_true")
parser.add_argument("--jobs", type=int, default=2)

args = parser.parse_args()

random.seed(args.seed)

def process_single_video(video_index:int, video_path:str, save_path:str):
    segmenter = VideoSegmenterGen(video_path)
    # Video info
    video_id = segmenter.get_video_id()
    video_class = segmenter.get_video_class()

    video_frames = segmenter.get_frames_generator()

    video_segment = 0
    while True:  # Iterate over segments
        filename = f"video{video_id}_{video_class}_{video_segment*args.window_size}-{(video_segment+1)*args.window_size}"
        segment_folder = Path(save_path, filename)
        segment_folder.mkdir(exist_ok=True, parents=True)

        fifo_frames = FifoFrames()
        is_video_ends = False
        frames = video_frames
        for i in range(args.window_size):
            print(
                f"{video_index}/{len(videos_paths)} Processing {video_id} segment: {video_segment} frame {i}/{args.window_size}",
                end="\r" if video_index + 1 < len(videos_paths) else "\n",
            )
            try:                      
                frame = next(frames)
                fifo_frames.append(frame)
            except StopIteration:
                is_video_ends = True
                if len(fifo_frames)>0:
                    frames = fifo_frames.reverse_loop_generator()
                else:
                    shutil.rmtree(segment_folder)
                    break


            cv2.imwrite(os.path.join(segment_folder, f"frame_{i}.png"), frame)
        if is_video_ends:
            break
        video_segment += 1

def process_subset(subset_path: str, save_path: str):
    pool = ThreadPool(args.jobs)      
    for video_index, video_path in enumerate(subset_path):
       pool.apply_async(process_single_video, (video_index, video_path, save_path))
    pool.close()
    pool.join()


videos_paths = glob.glob(os.path.join(args.video_path, "*", "*.mkv"))
if args.validation_split == 0:
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    process_subset(videos_paths, output_path)
else:
    random.shuffle(videos_paths)
    amount_of_videos = len(videos_paths)
    amount_of_validation_videos = int(args.validation_split * amount_of_videos)

    validation_paths = videos_paths[:amount_of_validation_videos]
    train_paths = videos_paths[amount_of_validation_videos:]

    print(f"{len(train_paths)} videos are selected for the training")
    print(f"{len(validation_paths)} videos are selected for the validation")

    train_windowed_path = Path(args.output_path, "train")
    val_windowed_path = Path(args.output_path, "val")
    train_windowed_path.mkdir(exist_ok=True, parents=True)
    val_windowed_path.mkdir(exist_ok=True, parents=True)

    process_subset(validation_paths, val_windowed_path)
    process_subset(train_paths, train_windowed_path)


