import argparse
import os
import re
import random
import shutil
from pathlib import Path
'''
Split the segment between validation and training set
'''
parser = argparse.ArgumentParser()
parser.add_argument(
    "--segment_path",
    type=str,
    help="Folder containing the video segments",
    required=True,
)
parser.add_argument(
    "--validation_split", type=float, default=0.3, help="Validation split"
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Folder where the output videos are stored",
    required=True,
)
args = parser.parse_args()
random.seed(1234)

# Load all segments in a organized data structure
def get_video_id(folder_name:str) -> int:
    videoX = folder_name.split("_")[0]
    id = re.findall(r'\d+', videoX)[0]
    return int(id)

segment_dict={"infested":{}, "free":{}}
segments = os.listdir(args.segment_path)
random.shuffle(segments)
for segment in segments:
    video_id = get_video_id(segment)
    if "infested" in segment:
        if str(video_id) not in segment_dict["infested"].keys():
            segment_dict["infested"][str(video_id)]=[]
        segment_dict["infested"][str(video_id)].append(os.path.join(args.segment_path, segment))
    elif "free" in segment:
        if str(video_id) not in segment_dict["free"].keys():
            segment_dict["free"][str(video_id)]=[]
        segment_dict["free"][str(video_id)].append(os.path.join(args.segment_path, segment))

# Count the amount of segments
amount = 0
for video_id in segment_dict["infested"]:
    amount += len(segment_dict["infested"][video_id])
for video_id in segment_dict["free"]:
    amount += len(segment_dict["free"][video_id])
print(f"Amount of segments {amount}")

# Take validation set wit the same amount between infested and free
validation_size = int(amount * args.validation_split)
print(f"Target validation size: {validation_size} | Target training size: {amount-validation_size}")

best_config:dict = {"free":{}, "infested":{}}
def get_taken_segments(subset:str) -> int:
    taken:int = 0
    for video in best_config[subset].keys():
        taken += len(best_config[subset][video])
    return taken


# Takes N/2 free segments
for video in segment_dict["free"].keys():
    segments = segment_dict["free"][video]
    print(f"Free: {get_taken_segments('free')} - Infested: {get_taken_segments('infested')}")
    if len(segments) < validation_size//2 - get_taken_segments("free"):
        best_config["free"][video] = segments

# Takes same amount of infested segments
for video in segment_dict["infested"].keys():
    segments = segment_dict["infested"][video]
    print(f"Free: {get_taken_segments('free')} - Infested: {get_taken_segments('infested')} | {len(segments)}")
    if len(segments) <= get_taken_segments("free") - get_taken_segments('infested'):
        best_config["infested"][video] = segments

# Takes the remaining as train set
validation_set = best_config.copy()
train_set:dict = {"free":{}, "infested":{}}
for video in segment_dict["free"]:
    if video not in validation_set["free"].keys():
        train_set["free"][video]=segment_dict["free"][video]
for video in segment_dict["infested"]:
    if video not in validation_set["infested"].keys():
        train_set["infested"][video]=segment_dict["infested"][video]

# Check if a video id is in both validation and training set
assert len(set(validation_set["free"].keys()).intersection(set(train_set["free"].keys())))==0
assert len(set(validation_set["infested"].keys()).intersection(set(train_set["infested"].keys())))==0

# Copy to output folder
train_path = Path(args.output_path,"train")
val_path = Path(args.output_path,"val")
train_path.mkdir(exist_ok=True, parents=True)
val_path.mkdir(exist_ok=True, parents=True)

for video in validation_set["free"].keys():
    print(f"Val: Processing free video: {video}")
    for segment in validation_set["free"][video]:
        segment_name = os.path.basename(segment)
        shutil.copytree(segment, os.path.join(val_path,segment_name))
for video in validation_set["infested"].keys():
    print(f"Val: Processing infested video: {video}")
    for segment in validation_set["infested"][video]:
        segment_name = os.path.basename(segment)
        shutil.copytree(segment, os.path.join(val_path,segment_name))

for video in train_set["free"].keys():
    print(f"Train: Processing free video: {video}")
    for segment in train_set["free"][video]:
        segment_name = os.path.basename(segment)
        shutil.copytree(segment, os.path.join(train_path,segment_name))
for video in train_set["infested"].keys():
    print(f"Train: Processing infested video: {video}")
    for segment in train_set["infested"][video]:
        segment_name = os.path.basename(segment)
        shutil.copytree(segment, os.path.join(train_path,segment_name))

