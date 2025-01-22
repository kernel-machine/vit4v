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
            img = img.resize([224, 224])
            img = torchvision.transforms.functional.pil_to_tensor(img)
            return img

        images = torch.stack(list(map(load_img, frames_paths)))
        chain = self.create_augment_chain(self.use_augmentation)
        images = torch.stack([chain(img) for img in images]).squeeze(1)

        return images, label

    def create_augment_chain(
        self, enable_augmentation: bool
    ) -> torchvision.transforms.Compose:
        if enable_augmentation:
            def random_crop(img) -> torch.Tensor:
                original_size = img.shape[1]
                new_size = random.randrange(int(original_size*0.7),original_size)
                img = torchvision.transforms.functional.center_crop(img, new_size)
                img = torchvision.transforms.functional.resize(img, original_size)
                return img
            
            return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda img: torchvision.transforms.functional.rotate(
                            img, float(random.randint(-180, 180))
                        )
                    ),
                    torchvision.transforms.Lambda(random_crop),
                    torchvision.transforms.Lambda(lambda img: img if random.random()>0.5 else torchvision.transforms.functional.hflip(img)),
                    torchvision.transforms.Lambda(lambda img: img if random.random()>0.5 else torchvision.transforms.functional.vflip(img)),
                    torchvision.transforms.ColorJitter(
                        brightness=float(random.randint(30, 80)) / 100,
                        saturation=float(random.randint(30, 80)) / 100,
                        hue=float(random.randint(0, 30)) / 100,
                        contrast=float(random.randint(30, 80)) / 100,
                    ),
                    torchvision.transforms.Lambda(
                        lambda img: self.image_processor(img, return_tensors="pt")[
                            "pixel_values"
                        ].squeeze(0)
                    ),
                ]
            )
        else:
            return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda img: self.image_processor(img, return_tensors="pt")[
                            "pixel_values"
                        ].squeeze(0)
                    )
                ]
            )
