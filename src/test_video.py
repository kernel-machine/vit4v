import torch
import argparse
import os
from lib.dataset.VideoSegmenter import VideoSegmenter
from lib.train.model_vivit import ModelVivit
from lib.train.model import MyModel
from PIL import Image
from torchvision import transforms
from lib.dataset.frame_buffer import FrameBuffer
import torchvision
import glob
from lib.validation_metric import ValidationMetrics
import time
import logging

movinet_found = True
try:
    from movinets import MoViNet
    from movinets.config import _C
except ModuleNotFoundError:
    movinet_found = False


args = argparse.ArgumentParser(description="Test a model on the orignal videos cropping the bee in realtime")
args.add_argument("--model", type=str, required=True, help="Path to weights")
args.add_argument("--video", type=str, required=True, help="Path to the video to process")
args.add_argument("--window_size", type=int, default=16, help="Size of the temporal window to process")
args = args.parse_args()

if args.export:
    import torch_tensorrt


def process_video(model: torch.nn.Module, video_path: str, window_size: int, device: torch.device, model_resolution:int, image_processing: callable = None) -> list[bool]:
    vs = VideoSegmenter(video_path, output_size=224)
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
    frame_buffer = FrameBuffer(window_size)
    avg_inference_time:tuple(float,int) = (0.0,0)
    pre_proc_times = []
    while True:
        start_frame_time = time.time()
        try:
            frame = next(frames)
        except StopIteration:
            break
        frame = Image.fromarray(frame).convert("RGB")
        frame = frame.resize([model_resolution, model_resolution])
        frame = torchvision.transforms.functional.pil_to_tensor(frame)
        frame_buffer.append(frame)
        pre_proc_times.append(time.time()-start_frame_time)

    for segment in frame_buffer.get_segments():
        with torch.no_grad():
            tensor_images = [f(img) for img in segment]
            tensor_images = torch.stack(tensor_images)
            if "movinet" in args.model:
                tensor_images = tensor_images.unsqueeze(0).permute(0,2,1,3,4).float()
            else:
                tensor_images = tensor_images.permute(1, 0, 2, 3, 4)
            #if "movinet" in args.model:
            #    tensor_images = tensor_images.permute(0,2,1,3,4)
            tensor_images = tensor_images.to(device)
            start_time = time.time()
            torch.cuda.synchronize()
            prediction_logits = model(tensor_images)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            avg_inference_time = (avg_inference_time[0]+inference_time, avg_inference_time[1]+1)
            predicted_classes = torch.sigmoid(prediction_logits).round().flatten()
            predictions.append(bool(predicted_classes.cpu()))

    return predictions, 0 if avg_inference_time[1]==0 else avg_inference_time[0]/avg_inference_time[1], sum(pre_proc_times)/len(pre_proc_times)

RESOLUTION = 0
if "vivit" in args.model:
    folder_name = os.path.basename(os.path.dirname(args.model))
    vivit_layer = int(folder_name.split("_")[1])
    if vivit_layer==12:
        vivit_layer=0
    model:ModelVivit = ModelVivit(hidden_layers=vivit_layer)
    auto_processing = model.get_image_processor()
    RESOLUTION = 224
    model = torch.nn.DataParallel(model)
    dummy_input = torch.randn(1, 32 , 3, RESOLUTION, RESOLUTION)
elif "movinet_a1" in args.model and movinet_found:
    model = MoViNet(_C.MODEL.MoViNetA1, causal = False, pretrained = True )
    model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
    auto_processing = None
    RESOLUTION = 172
    dummy_input = torch.randn(1, 3 , 32, RESOLUTION, RESOLUTION)
elif "movinet_a2" in args.model and movinet_found:
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

if args.export:
    model = model.module.eval()
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=[dummy_input])
    torch_tensorrt.save(trt_gm, "trt.ep", inputs=[dummy_input])
    exit(0)



torch.cuda.empty_cache()
a = torch.cuda.memory_allocated()
model.to(device)
with torch.no_grad():
    model(dummy_input)
torch.cuda.synchronize()
b = torch.cuda.memory_allocated()

dir_path = os.path.dirname(args.model)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir_path,'performance.log'), level=logging.INFO, filemode='w')
logging.getLogger().addHandler(logging.StreamHandler())

mem_bytes = b-a
mem_mb = mem_bytes/1024**2
logger.info(f"Memory allocated by the model {mem_bytes} Bytes -> {mem_mb:-2f} MB")


if os.path.isfile(args.video):
    process_video(model, args.video, window_size=args.window_size, device=device, model_resolution=RESOLUTION, image_processing=auto_processing)
else:
    print("Evaluating positive samples")
    def p(video:str) -> bool:
        predictions, avg_time, pre_proc_time = process_video(model, video, window_size=args.window_size, device=device, model_resolution=RESOLUTION, image_processing=auto_processing)
        folder_name = os.path.normpath(video).split(os.sep)[-2]
        print(f"{folder_name} {os.path.basename(video)} -> predictions: {predictions}, inf time: {avg_time}, pre-processing:  {pre_proc_time}")
        return predictions, avg_time
    most_common = ValidationMetrics()
    true_inside = ValidationMetrics()
    seg_vm = ValidationMetrics()
    vm_top_conf = ValidationMetrics()
    no_seg = 0
    times = []
    
    for video in glob.glob(os.path.join(args.video,"varroa_infested","*.mkv")):
        seg_preds, avg_time = p(video)        
        if len(seg_preds)==0:
            no_seg+=1
        for seg_pred in seg_preds:
            seg_vm.add_prediction(seg_pred,True)
        if len(seg_preds)>0:
            most_common.add_prediction(max(set(seg_preds), key=seg_preds.count),True)
            true_inside.add_prediction(True in seg_preds, True)
            times.append(avg_time)
    for video in glob.glob(os.path.join(args.video,"varroa_free","*.mkv")):
        seg_preds, avg_time = p(video)
        if len(seg_preds)==0:
            no_seg+=1
        for seg_pred in seg_preds:
            seg_vm.add_prediction(seg_pred,False)
        if len(seg_preds)>0:
            most_common.add_prediction(max(set(seg_preds), key=seg_preds.count),False)
            true_inside.add_prediction(True in seg_preds, False)
            times.append(avg_time)

    logger.info(f"Most common {most_common}")
    logger.info(f"True inside {true_inside}")
    logger.info(f"Per segment: {seg_vm}")
    logger.info(f"Video without segments {no_seg}")
    logger.info(f"Times {sum(times)/len(times)}")
    # cm = most_common.get_confusion_matrix()
    # cm.figure_.savefig(os.path.join(dir_path,"confusion_matrix.png"))