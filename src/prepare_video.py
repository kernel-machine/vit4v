import argparse
import glob
import os
import re
import shutil
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--segment_folder")
parser.add_argument("--raw_videos")
parser.add_argument("--dest")


args = parser.parse_args()
ids_infested = set()
ids_free = set()

def get_id_of_folder(folder_name) -> int:
    videoX = folder_name.split("_")[0]
    id = re.findall(r'\d+', videoX)[0]
    return int(id)

raw_videos_infested = glob.glob(os.path.join(args.raw_videos,"varroa_infested","*.mkv"))
raw_videos_free = glob.glob(os.path.join(args.raw_videos,"varroa_free","*.mkv"))
Path(args.dest,"varroa_infested").mkdir(exist_ok=True, parents=True)
Path(args.dest,"varroa_free").mkdir(exist_ok=True, parents=True)

input_videos = 0
copied_videos = 0
for video_folder in glob.glob(os.path.join(args.segment_folder,"*")):
    basename = os.path.basename(video_folder)
    segment_video_id = get_id_of_folder(basename)
    input_videos+=1
    if "infested" in basename:
        for raw_video in raw_videos_infested:
            raw_id = os.path.basename(raw_video).split(" ")[0]
            raw_id = int(raw_id)
            if raw_id == segment_video_id:
                dest_file = os.path.join(args.dest,"varroa_infested",f"{raw_id}_infested.mkv")
                print(f"Copying {raw_video} into {dest_file}")
                shutil.copy(raw_video,dest_file)
                copied_videos+=1
                break
    else:
        for raw_video in raw_videos_free:
            raw_id = os.path.basename(raw_video).split(" ")[0]
            raw_id = int(raw_id)
            if raw_id == segment_video_id:
                dest_file = os.path.join(args.dest,"varroa_free",f"{raw_id}_free.mkv")
                print(f"Copying {raw_video} into {dest_file}")
                shutil.copy(raw_video,dest_file)
                copied_videos+=1
                break

print(f"Input videos {input_videos} -> copied videos {copied_videos}")