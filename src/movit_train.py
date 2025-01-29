from movinets import MoViNet
from movinets.config import _C
import torch.nn.functional as F
import torch
from lib.train.dataset_loader import VarroaDataset
import torchvision.transforms as transforms
import torch
import torchvision
import glob
import numpy as np
import os
import random
from PIL import Image
import torchvision.transforms.functional
import torchvision.transforms.v2
from transformers import AutoImageProcessor
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--no_aug", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--devices", type=str, default="0,1,2,3")
parser.add_argument("--model", type=str)

args = parser.parse_args()

class VarroaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        videos_path: str,
        image_processor: callable = None,
        seed: int = 1234,
        use_augmentation: bool = False,
    ) -> None:
        random.seed(seed)
        self.videos_base_path = videos_path
        self.videos_paths = os.listdir(videos_path)
        self.videos_paths.sort()
        random.shuffle(self.videos_paths)
        self.image_processor = image_processor
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.videos_paths)
    
    def varroa_free_count(self):
        return sum(list(map(lambda x:"varroa_free" in x, self.videos_paths)))
    
    def varroa_infested_count(self):
        return sum(list(map(lambda x:"varroa_infested" in x, self.videos_paths)))
    
    def balance(self): #Remove elements from the lower dataset to balance the dataset
        while self.varroa_free_count() > self.varroa_infested_count():
            for index, item in enumerate(self.videos_paths):
                if "varroa_free" in item:
                    self.videos_paths.pop(index)
                    break
        while self.varroa_free_count() < self.varroa_infested_count():
            for index, item in enumerate(self.videos_paths):
                if "varroa_infested" in item:
                    self.videos_paths.pop(index)
                    break

    def __getitem__(self, index) -> any:
        video_path = self.videos_paths[index]
        label = 1 if "infested" in video_path else 0
        segment_path = os.path.join(self.videos_base_path, video_path)
        frames_paths = os.listdir(segment_path)

        def load_img(frame_path: str) -> torch.Tensor:
            absolute_path = os.path.join(segment_path, frame_path)
            img = Image.open(absolute_path).convert("RGB")
            img = img.resize([172, 172])
            img = torchvision.transforms.functional.pil_to_tensor(img)
            return img

        images = torch.stack(list(map(load_img, frames_paths)))
        #chain = self.create_augment_chain(self.use_augmentation)
        #images = torch.stack([chain(img) for img in images]).squeeze(1)
        images = torch.permute(images, [1,0,2,3])

        return images, label
    
    def create_augment_chain(
        self, enable_augmentation: bool
        ) -> torchvision.transforms.Compose:
        return torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
        ])
            

    
# Load dataset
val_ds = VarroaDataset(
    os.path.join(args.dataset, "val"),
    image_processor=False,
    use_augmentation=False,
)
train_ds = VarroaDataset(
    os.path.join(args.dataset, "train"),
    image_processor=False,
    use_augmentation=(not args.no_aug),
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
    val_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
)

model = MoViNet(_C.MODEL.MoViNetA0, causal = True, pretrained = True )
model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for image,label in train_dataloader:
    image = image.float().cuda()
    label = label.cuda()
    logits = model(image)
    print(logits)

