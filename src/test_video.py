import imp
import torch
import argparse
import os
from lib.dataset.VideoSegmenter import VideoSegmenter
from lib.train.model_vivit import ModelVivit
from lib.train.model import MyModel
from PIL import Image
from torchvision import transforms
from lib.dataset.fifo_frames import FifoFrames
import torchvision
import glob
from lib.validation_metric import ValidationMetrics
import time
args = argparse.ArgumentParser()
from movinets import MoViNet
from movinets.config import _C

args.add_argument("--model", type=str, required=True, help="Path to weights")
args.add_argument("--video", type=str, required=True, help="Path to the video to process")
args.add_argument("--window_size", type=int, default=16)
args = args.parse_args()


def process_video(model: torch.nn.Module, video_path: str, window_size: int, device: torch.device, model_resolution:int, image_processing: callable = None) -> list[bool]:
    vs = VideoSegmenter(video_path)
    frames = vs.get_frames()
    f = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x:x) if image_processing is None else torchvision.transforms.Lambda(
                        lambda img: image_processing(img, return_tensors="pt")[
                            "pixel_values"
                        ].squeeze(0)
                    ),
                ]
            )
    predictions = []
    frames_buffer = []
    avg_inference_time:tuple(float,int) = (0.0,0)
    while True:
        try:
            frame = next(frames)
        except StopIteration:
            break
        frame = Image.fromarray(frame).convert("RGB")
        frame = frame.resize([model_resolution, model_resolution])
        frame = torchvision.transforms.functional.pil_to_tensor(frame)
        frames_buffer.append(frame)

        if len(frames_buffer) == window_size:
            with torch.no_grad():
                tensor_images = [f(img) for img in frames_buffer]
                tensor_images = torch.stack(tensor_images)
                if "movinet" in args.model:
                    tensor_images = tensor_images.unsqueeze(0).permute(0,2,1,3,4).float()
                else:
                    tensor_images = tensor_images.permute(1, 0, 2, 3, 4)
                #if "movinet" in args.model:
                #    tensor_images = tensor_images.permute(0,2,1,3,4)
                tensor_images = tensor_images.to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                prediction_logits = model(tensor_images)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = end_time - start_time
                avg_inference_time = (avg_inference_time[0]+inference_time, avg_inference_time[1]+1)
                predicted_classes = torch.sigmoid(prediction_logits).round().flatten()
                predictions.append(bool(predicted_classes.cpu()))
                frames_buffer.clear()
    return predictions, 0 if avg_inference_time[1]==0 else avg_inference_time[0]/avg_inference_time[1]

RESOLUTION = 0
if "vivit" in args.model:
    model:ModelVivit = ModelVivit()
    auto_processing = model.get_image_processor()
    RESOLUTION = 224
    model = torch.nn.DataParallel(model)
    dummy_input = torch.randn(1, 32 , 3, RESOLUTION, RESOLUTION)
elif "movinet_a1" in args.model:
    model = MoViNet(_C.MODEL.MoViNetA1, causal = False, pretrained = True )
    model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
    auto_processing = None
    RESOLUTION = 172
    dummy_input = torch.randn(1, 3 , 32, RESOLUTION, RESOLUTION)
elif "movinet_a2" in args.model:
    model = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True )
    model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
    auto_processing = None
    RESOLUTION = 224
    dummy_input = torch.randn(1, 3 , 32, RESOLUTION, RESOLUTION)
else:
    print("loading")
    model:MyModel = MyModel()
    auto_processing = model.get_image_processor()
    RESOLUTION = 224
    model = torch.nn.DataParallel(model)
    dummy_input = torch.randn(1, 32 , 3, RESOLUTION, RESOLUTION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = dummy_input.to(device)
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))

torch.cuda.empty_cache()
a = torch.cuda.memory_reserved()
model.to(device)
with torch.no_grad():
    model(dummy_input)
torch.cuda.synchronize()
b = torch.cuda.memory_reserved()

mem_bytes = b-a
mem_mb = mem_bytes/1024**2
print(f"Memory allocated by the model {mem_bytes} Bytes -> {mem_mb:-2f} MB")


if os.path.isfile(args.video):
    process_video(model, args.video, window_size=args.window_size, device=device, model_resolution=RESOLUTION, image_processing=auto_processing)
else:
    print("Evaluating positive samples")
    def p(video:str) -> bool:
        predictions, avg_time = process_video(model, video, window_size=args.window_size, device=device, model_resolution=RESOLUTION, image_processing=auto_processing)
        folder_name = os.path.normpath(video).split(os.sep)[-2]
        try:
            most_common = max(set(predictions), key=predictions.count)
        except ValueError:
            most_common = False
        print(f"{folder_name} {os.path.basename(video)} -> predictions: {predictions} -> Verdict: {most_common}, time: {avg_time}")
        return most_common, predictions
    vm = ValidationMetrics()
    seg_vm = ValidationMetrics()
    vm_top_conf = ValidationMetrics()
    for video in glob.glob(os.path.join(args.video,"varroa_infested","*.mkv")):
        prediction, seg_preds = p(video)
        for seg_pred in seg_preds:
            seg_vm.add_prediction(seg_pred,True)
        if len(seg_preds)>0:
            vm.add_prediction(prediction,True)
    for video in glob.glob(os.path.join(args.video,"varroa_free","*.mkv")):
        prediction, seg_preds = p(video)
        for seg_pred in seg_preds:
            seg_vm.add_prediction(seg_pred,False)
        if len(seg_preds)>0:
            vm.add_prediction(prediction,False)

    print(f"Per Video: F1: {vm.get_f1()} | Acc: {vm.get_accuracy()}")
    tp, fp, tn, fn = vm.get_metrics()
    print(f"Per Video: TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}")
    print(f"Per Segment: F1: {seg_vm.get_f1()} | Acc: {seg_vm.get_accuracy()}")
    cm = vm.get_confusion_matrix()
    cm.figure_.savefig("confusion_matrix.png")