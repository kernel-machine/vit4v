import cv2
import argparse
import os
import numpy as np
from pathlib import Path
from lib.dataset.unical_parser import UniCalParser, Camera, VideoClass
from collections.abc import Generator
import time

MIN_AREA_THRESHOLD = 5000
BOX_SIZE = 920

class VideoSegmenter:
    def __init__(self, video_path:str, output_size=1000, show_debug:bool=False) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.output_size = output_size
        self.show_debug = show_debug
        if self.show_debug:
            video_id = os.path.basename(self.video_path).split(" ")[0]
            video_name = f"video_{video_id}_{int(time.time_ns()/1000)}_debug.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vwriter = cv2.VideoWriter(video_name, fourcc, 20, (2*output_size, output_size))
            self.vwriter.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

    def get_frames(self) -> Generator[cv2.Mat]:
        background = None
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = frame[0:-180][:]
                (frame_h,frame_w) = frame.shape[:2]

                # Get Saturation and Value mask
                frame_blur = cv2.GaussianBlur(frame, (7,7), 10)
                frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
                saturation_channel = cv2.normalize(frame_hsv[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
                _, mask_saturation = cv2.threshold(saturation_channel, 50, 255, cv2.THRESH_BINARY) # For saturation
                value_channel = cv2.normalize(frame_hsv[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
                _, mask_value = cv2.threshold(value_channel, 60, 255, cv2.THRESH_BINARY_INV)
                mask = cv2.bitwise_and(mask_saturation, mask_value)
                if background is None:
                    background = mask
                else:
                    motion_mask = cv2.absdiff(background, mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                    inverted_motion_mask = cv2.bitwise_not(motion_mask)
                    contours, _ = cv2.findContours(inverted_motion_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    threshold_area = 250

                    # Fill small holes to remove noise
                    for c in contours:
                        if cv2.contourArea(c) < threshold_area:
                            motion_mask = cv2.drawContours(motion_mask, [c], -1, 0, cv2.FILLED)
                    mask = cv2.bitwise_and(mask, motion_mask)
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        contours = list(filter(lambda x:cv2.boundingRect(x)[2]>150, contours)) #Remove small holes
                        if len(contours) > 0:
                            areas = list(map(lambda x:cv2.contourArea(x), contours))
                            max_index = np.argmax(areas)
                            if areas[max_index] > MIN_AREA_THRESHOLD: #Take the largest hole
                                c = contours[max_index]
                                x, y, w, h = cv2.boundingRect(c)
                                center = (x+(w//2),y+(h//2))
                                top_left = (center[0]-BOX_SIZE//2,center[1]-BOX_SIZE//2)
                                bottom_right = (center[0]+BOX_SIZE//2,center[1]+BOX_SIZE//2)
                                if bottom_right[1]>frame_h:
                                    out_size = bottom_right[1]-frame_h
                                    top_left = (top_left[0], top_left[1]-out_size)
                                if top_left[1] < 0:
                                    bottom_right = (bottom_right[0], bottom_right[1]-top_left[1])
                                    top_left = (top_left[0], 0)                        
                                x1, y1 = top_left
                                x2, y2 = bottom_right
                                x1, y1 = max(x1, 0), max(y1, 0)
                                x2, y2 = min(x2, frame_w), min(y2, frame_h)
                                cropped_frame = frame[y1:y2, x1:x2]
                                yield cv2.resize(cropped_frame, (self.output_size, self.output_size))
                                if self.show_debug:
                                    frame = cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 20)
                    if self.show_debug and self.vwriter is not None:
                        frame = cv2.resize(frame, (2*self.output_size, self.output_size))
                        self.vwriter.write(frame)
            else:
                return
    def get_video_length(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_video_id(self) -> int:
        return int(os.path.basename(self.video_path).split(" ")[0])

    def get_video_class(self) -> str:
        return self.video_path.split(os.sep)[-2]
    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Input video to segmet")
    parser.add_argument("--save_out", action="store_true",default=False)
    parser.add_argument("--disable_video", action="store_true",default=False)
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--show_mask", action="store_true", default=False)
    parser.add_argument("--label_file", type=str, default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    backSub = cv2.createBackgroundSubtractorMOG2()
    frames = []
    frame_index = 0
    background_update_rate = 10
    video_id = os.path.basename(args.video).split(" ")[0]
    class_name = args.video.split(os.sep)[-2]

    print(f"Processing {class_name} video_{video_id}")

    vwriter = None
    if args.save_out:
        video_name = f"{class_name}_{video_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vwriter = cv2.VideoWriter(video_name, fourcc, 20, (1200,367)) # TODO ADJUST SIZE
        vwriter.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

    if args.out_folder:
        Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    unical_label:UniCalParser = None
    if args.label_file is not None:
        unical_label = UniCalParser(args.label_file)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = frame[0:-180][:]
            (frame_h,frame_w) = frame.shape[:2]
            frame_blur = cv2.GaussianBlur(frame, (7,7), 10)
            frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

            lr = np.array([40, 255, 40])
            ur = np.array([60, 255, 255])
            saturation_channel = cv2.normalize(frame_hsv[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
            _, mask_saturation = cv2.threshold(saturation_channel, 50, 255, cv2.THRESH_BINARY) # For saturation
            value_channel = cv2.normalize(frame_hsv[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
            _, mask_value = cv2.threshold(value_channel, 60, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.bitwise_and(mask_saturation, mask_value)
            if frame_index == 0:
                background = mask
            else:
                motion_mask = cv2.absdiff(background, mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                inverted_motion_mask = cv2.bitwise_not(motion_mask)
                contours, _ = cv2.findContours(inverted_motion_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                threshold_area = 250
                # Riempie solo i buchi piccoli
                for c in contours:
                    if cv2.contourArea(c) < threshold_area:
                        motion_mask = cv2.drawContours(motion_mask, [c], -1, 0, cv2.FILLED)
                mask = cv2.bitwise_and(mask, motion_mask)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    contours = list(filter(lambda x:cv2.boundingRect(x)[2]>150, contours))
                    if len(contours) > 0:
                        areas = list(map(lambda x:cv2.contourArea(x), contours))
                        max_index = np.argmax(areas)
                        if areas[max_index] > 20000:
                            c = contours[max_index]
                            x, y, w, h = cv2.boundingRect(c)
                            box_size = 920
                            center = (x+(w//2),y+(h//2))
                            top_left = (center[0]-box_size//2,center[1]-box_size//2)
                            bottom_right = (center[0]+box_size//2,center[1]+box_size//2)
                            camera_position = Camera.TOP if x < frame_w/2 else Camera.BOTTOM
                            video_class = VideoClass.VARROA_INFESTED if "infested" in class_name else VideoClass.VARROA_FREE
                            if unical_label is not None:
                                label_found, varroa_visible = unical_label.get_frame_info(camera_position, video_class, video_id, frame_index)
                            if unical_label is None or label_found:
                                if bottom_right[1]>frame_h:
                                    out_size = bottom_right[1]-frame_h
                                    top_left = (top_left[0], top_left[1]-out_size)
                                if top_left[1] < 0:
                                    bottom_right = (bottom_right[0], bottom_right[1]-top_left[1])
                                    top_left = (top_left[0], 0)                        
                                if args.out_folder is not None:
                                    x1, y1 = top_left
                                    x2, y2 = bottom_right
                                    x1, y1 = max(x1, 0), max(y1, 0)
                                    x2, y2 = min(x2, frame_w), min(y2, frame_h)
                                    cropped_frame = frame[y1:y2, x1:x2]
                                    filename = os.path.join(args.out_folder,f"video{video_id}_frame{frame_index}_{class_name}_v{1 if varroa_visible else 0}.png")
                                    cv2.imwrite(filename, cropped_frame)
                                if label_found and not varroa_visible:
                                    color = (0,255,0)
                                elif label_found and varroa_visible:
                                    color = (0,0,255)
                                else:
                                    color = (255,0,0)
                                frame = cv2.rectangle(frame, top_left, bottom_right, color, 20)
                            # cv2.imshow('Motion Mask', frame)
                            # cv2.waitKey(1000//60)

                if vwriter is not None:
                    vwriter.write(frame)
                
                if args.show_mask:
                    mask_saturation = cv2.cvtColor(mask_saturation, cv2.COLOR_GRAY2BGR)
                    mask_value = cv2.cvtColor(mask_value, cv2.COLOR_GRAY2BGR)
                    motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    frame = np.vstack([mask_saturation, mask_value, motion_mask, mask,frame])

                if not args.disable_video:
                    crop_factor = 0.1
                    (frame_h,frame_w) = frame.shape[:2]
                    new_width = int(crop_factor*frame_w)
                    new_height = int(crop_factor*frame_h)
                    frame = cv2.resize(frame, (new_width, new_height))
                    frame = cv2.putText(frame,f"Frame: {frame_index}",(10,new_height-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),1,cv2.LINE_AA)
                    cv2.imshow('Motion Mask', frame)
                    cv2.waitKey(1000//120)
            frame_index+=1
        else:
            break

    if vwriter is not None:
        vwriter.release()
    cap.release()
    cv2.destroyAllWindows()

