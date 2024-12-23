import torchvision.transforms.functional
import transformers
import numpy as np
from lib.train.dataset_loader import VarroaDataset
from lib.train.model import MyModel
from lib.validation_metric import ValidationMetrics
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
import os
import random
import torchvision
import json

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--no_aug", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--devices", type=str, default="0,1,2,3")
parser.add_argument("--pre_trained_model", type=str, required=True) # facebook/timesformer-base-finetuned-ssv2
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Load model
pretrained_model = args.pre_trained_model
model:MyModel = MyModel(base_model=pretrained_model)
auto_processing = model.get_image_processor()

devices = args.devices
devices = devices.split(",")
devices = list(map(lambda x:int(x), devices))
device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=devices).cuda()
model.load_state_dict(torch.load(args.model, weights_only=True))

# Load dataset
val_ds = VarroaDataset(
    os.path.join(args.dataset, "val"),
    image_processor=auto_processing,
    use_augmentation=False,
)
train_ds = VarroaDataset(
    os.path.join(args.dataset, "train"),
    image_processor=auto_processing,
    use_augmentation=(not args.no_aug),
)

val_ds.balance()
print(f"Validation set - Free:{val_ds.varroa_free_count()} Infested:{val_ds.varroa_infested_count()}")
print(f"Train set - Free:{train_ds.varroa_free_count()} Infested:{train_ds.varroa_infested_count()}")

val_dataloader = torch.utils.data.DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)

pos_weight = torch.tensor([train_ds.varroa_free_count()/train_ds.varroa_infested_count()]).cuda()
print(f"Weight positive: {pos_weight}")
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

vm = ValidationMetrics()
model.eval()
running_val_loss = 0.0
correct_val_predictions = 0
total_val_predictions = 0
with torch.no_grad():
    for i, (frames, labels) in enumerate(val_dataloader):
        frames = frames.cuda()
        out = model(frames)
        out = out.flatten()
        labels = labels.float().cuda()
        loss = loss_fn(out, labels)

        predicted_classes = torch.sigmoid(out).round()
        correct_val_predictions += (predicted_classes == labels).sum().item()
        total_val_predictions += labels.size(0)
        running_val_loss += loss.item()
        print(f"Batch {i} -> Prediction: {predicted_classes.cpu()}")
        print(f"Batch {i} -> Labels:     {labels.cpu()}")
        for j in range(len(predicted_classes)):
            prediction = predicted_classes[j].cpu()>0.5
            vm.add_prediction(bool(prediction), bool(labels[j]))

        current_accuracy = correct_val_predictions / total_val_predictions
        current_loss = running_val_loss / (i + 1)
        print(
            f"Validation Batch {i}/{len(val_dataloader)}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}",
            end="\r" if i + 1 < len(val_dataloader) else "\n",
        )

print(f"F1: {vm.get_f1()}")
tp, fp, tn, fn = vm.get_metrics()
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}")
cm = vm.get_confusion_matrix()
dir_path = os.path.dirname(args.model)
cm.figure_.savefig(os.path.join(dir_path,"confusion_matrix.png"))