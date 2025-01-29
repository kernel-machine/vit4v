import torchvision.transforms.functional
import transformers
import numpy as np
from lib.train.dataset_loader import VarroaDataset
from lib.train.model import MyModel
from lib.train.model_vivit import ModelVivit
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
import os
import random
import torchvision
import json
from movinets import MoViNet
from movinets.config import _C

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--no_aug", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--devices", type=str, default="0")
parser.add_argument("--pre_trained_model", type=str, default=None) # facebook/timesformer-base-finetuned-ssv2
parser.add_argument("--model", type=str)

args = parser.parse_args()

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

now = datetime.now()  # current date and time
date_time = now.strftime("%Y-%b-%d-%H:%M:%S")
comment = f"BS_{args.batch_size}_aug{args.no_aug}_lr{args.lr}_ds{os.path.basename(args.dataset)}"
log_dir = os.path.join("../runs", date_time)
writer = SummaryWriter(log_dir=log_dir)
with open(os.path.join(log_dir,"args.json"),"w") as f:
    f.write(json.dumps(vars(args)))
    f.close()

# Load model
pretrained_model = args.pre_trained_model
RESOLUTION = 0
if args.model == "vivit":
    model:ModelVivit = ModelVivit()
    writer.add_text("Model","ViVit")
    auto_processing = model.get_image_processor()
    RESOLUTION = 224
elif args.model == "movinet":
    model = MoViNet(_C.MODEL.MoViNetA1, causal = False, pretrained = True )
    model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
    auto_processing = None
    RESOLUTION = 172
else:
    model:MyModel = MyModel()
    writer.add_text("Model","TimeSformer")
    auto_processing = model.get_image_processor()
    RESOLUTION = 224

devices = args.devices
devices = devices.split(",")
devices = list(map(lambda x:int(x), devices))
torch.cuda.set_device(devices[0])
device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
model.to(device)
#model = torch.nn.DataParallel(model, device_ids=devices).cuda()
model.clean_activation_buffers()

# Load dataset
val_ds = VarroaDataset(
    os.path.join(args.dataset, "val"),
    image_processor=auto_processing,
    use_augmentation=False,
    format = "movinet"
)
train_ds = VarroaDataset(
    os.path.join(args.dataset, "train"),
    image_processor=auto_processing,
    use_augmentation=(not args.no_aug),
    format = "movinet"
)

val_ds.balance()
print(f"Validation set - Free:{val_ds.varroa_free_count()} Infested:{val_ds.varroa_infested_count()}")
print(f"Train set - Free:{train_ds.varroa_free_count()} Infested:{train_ds.varroa_infested_count()}")

train_dataloader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
)
val_dataloader = torch.utils.data.DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)

pos_weight = torch.tensor([train_ds.varroa_free_count()/train_ds.varroa_infested_count()]).cuda()
print(f"Weight positive: {pos_weight}")
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
random_values = {}

best_validation_loss = float("inf")
for epoch in range(args.epochs):
    model.train()

    running_train_loss = 0
    correct_train_predictions = 0
    total_train_predictions = 0
    # Train
    for i, (frames, labels) in enumerate(train_dataloader):
        # augment_chain = create_augment_chain(enable_augmentation=True)
        frames: torch.Tensor = frames.cuda()
        # frames = pre_process_batch(frames , augment_chain)
        labels = labels.float().cuda()

        optimizer.zero_grad()
        out = model(frames)
        # if args.model == "movinet":
        #     out = torch.nn.LogSoftmax(dim=1)(out)
        out = out.flatten()
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        if args.model == "movinet":
            model.clean_activation_buffers()
            optimizer.zero_grad()
        predicted_classes = torch.sigmoid(out).round()
        correct_train_predictions += (predicted_classes == labels).sum().item()
        total_train_predictions += labels.size(0)
        running_train_loss += loss.item()

        current_loss = running_train_loss / (i + 1)
        current_accuracy = correct_train_predictions / total_train_predictions

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Batch {i}/{len(train_dataloader)}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}",
            end="\r" if i + 1 < len(train_dataloader) else "\n",
        )

    epoch_train_loss = running_train_loss / len(train_dataloader)
    epoch_train_accuracy = correct_train_predictions / total_train_predictions
    writer.add_scalar("Loss/train", epoch_train_loss, epoch)
    writer.add_scalar("Acc/train", epoch_train_accuracy, epoch)
    writer.flush()

    if args.model == "movinet":
        model.clean_activation_buffers()
    # Validation
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

            current_accuracy = correct_val_predictions / total_val_predictions
            current_loss = running_val_loss / (i + 1)
            print(
                f"Validation Epoch {epoch + 1}/{args.epochs}, | Batch {i}/{len(val_dataloader)}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}",
                end="\r" if i + 1 < len(val_dataloader) else "\n",
            )

    epoch_val_loss = running_val_loss / len(val_dataloader)
    epoch_val_accuracy = correct_val_predictions / total_val_predictions
    writer.add_scalar("Loss/val", epoch_val_loss, epoch)
    writer.add_scalar("Acc/val", epoch_val_accuracy, epoch)
    writer.flush()

    if epoch_val_loss < best_validation_loss:
        best_validation_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(log_dir,"model.pth"))
        writer.add_text("Best epoch",f"Epoch: {epoch}")
