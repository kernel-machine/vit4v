import os
import argparse
import re
import shutil
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str)
parser.add_argument("--raw_videos",type=str)
parser.add_argument("--output_videos",type=str)
args = parser.parse_args()

varroa_free_videos = set()
varroa_infested_videos = set()

def get_video_id(folder_name):
    id = folder_name.split("_")[0]
    return int(re.findall(r'\d+', id)[0])

assert get_video_id("video11_bla")!=get_video_id("video111_bla")

for video in os.listdir(args.dataset):
    id = get_video_id(video)
    class_name = video.split("_")[2]
    assert f"video{id}_varroa" in video
    if class_name == "infested":
        varroa_infested_videos.add(id)
    else:
        varroa_free_videos.add(id)
    
print(varroa_infested_videos)
print(varroa_free_videos)

videos = glob.glob(os.path.join(args.raw_videos,"*","*.mkv"))
def get_src(id:str,class_name:str):
    for v in videos:
        folder = v.split(os.sep)[-2]
        if int(os.path.basename(v).split(" ")[0])==int(id) and class_name in folder:
            #print(f"id:{id} - {class_name} -> {v}")
            return v
    
def check_if_segment_present(id) -> bool:
    segments = glob.glob(os.path.join(args.dataset,"*"))
    for s in segments:
        if get_video_id(s)==id:
            #print(f"ID {id} -> {s}")
            return True
    return False

infested_folder = Path(args.output_videos,"varroa_infested")
infested_folder.mkdir(parents=True,exist_ok=True)
for id in varroa_infested_videos:
    src = get_src(id,"infested")
    dest = os.path.join(infested_folder, os.path.basename(src))
    shutil.copy(src,dest)
    print(f"{id} | Copying {src} -> {dest}")
    assert f"varroa_infested/{id} " in src

free_folder = Path(args.output_videos,"varroa_free")
free_folder.mkdir(parents=True,exist_ok=True)
for id in varroa_free_videos:
    src = get_src(id,"free")
    dest = os.path.join(free_folder, os.path.basename(src))
    shutil.copy(src,dest)
    print(f"{id} | Copying {src} -> {dest}")
    assert f"varroa_free/{id} " in src
