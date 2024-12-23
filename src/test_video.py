import torch
import argparse
import os
from lib.dataset.VideoSegmenter import VideoSegmenter
from lib.train.model import MyModel
from PIL import Image
from torchvision import transforms
from lib.dataset.fifo_frames import FifoFrames
import torchvision
import glob
from lib.validation_metric import ValidationMetrics
args = argparse.ArgumentParser()
args.add_argument("--model", type=str, required=True, help="Path to weights")
args.add_argument("--video", type=str, required=True, help="Path to the video to process")
args.add_argument("--window_size", type=int, default=16)
args = args.parse_args()


def process_video(model: MyModel, video_path: str, window_size: int, device: torch.device) -> list[bool]:
    vs = VideoSegmenter(video_path)
    frames = vs.get_frames()
    f = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: model.module.prepare_images(img)[
                    "pixel_values"
                ].squeeze(0)
            )
        ]
    )
    predictions = []
    frames_buffer = []
    while True:
        try:
            frame = next(frames)
        except StopIteration:
            break
        frame = Image.fromarray(frame).convert("RGB")
        frame = frame.resize([224, 224])
        frame = torchvision.transforms.functional.pil_to_tensor(frame)
        frames_buffer.append(frame)

        if len(frames_buffer) == window_size:
            with torch.no_grad():
                tensor_images = [f(img) for img in frames_buffer]
                tensor_images = torch.stack(tensor_images)
                tensor_images = tensor_images.permute(1, 0, 2, 3, 4)
                tensor_images = tensor_images.to(device)
                prediction_logits = model(tensor_images)
                predicted_classes = torch.sigmoid(prediction_logits).round().flatten()
                predictions.append(bool(predicted_classes.cpu()))
                frames_buffer.clear()
    return predictions


model: MyModel = MyModel(base_model="facebook/timesformer-base-finetuned-k400")
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(args.model, weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if os.path.isfile(args.video):
    process_video(model, args.video, window_size=args.window_size, device=device)
else:
    print("Evaluating positive samples")
    def p(video:str) -> bool:
        predictions = process_video(model, video, window_size=args.window_size, device=device)
        folder_name = os.path.normpath(video).split(os.sep)[-2]
        try:
            most_common = max(set(predictions), key=predictions.count)
        except ValueError:
            most_common = False
        print(f"{folder_name} {os.path.basename(video)} -> predictions: {predictions} -> Verdict: {most_common}")
        return most_common
    vm = ValidationMetrics()
    for video in glob.glob(os.path.join(args.video,"varroa_infested","*.mkv")):
        prediction = p(video)
        vm.add_prediction(prediction,True)
    for video in glob.glob(os.path.join(args.video,"varroa_free","*.mkv")):
        prediction = p(video)
        vm.add_prediction(prediction,False)

    print(f"F1: {vm.get_f1()}")
    tp, fp, tn, fn = vm.get_metrics()
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}")
    cm = vm.get_confusion_matrix()
    cm.figure_.savefig("confusion_matrix.png")