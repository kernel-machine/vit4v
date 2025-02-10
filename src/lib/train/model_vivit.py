from typing import Optional, Union, Tuple
from transformers import VivitModel,VivitImageProcessor, VivitConfig,VivitForVideoClassification
import torch
    
class ModelVivit(torch.nn.Module):
    def __init__(self, base_model:str = "google/vivit-b-16x2-kinetics400", hidden_layers:int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "google/vivit-b-16x2-kinetics400"
        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        num_classes = 1
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes, bias=True)
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

        if hidden_layers > 0:
            self.model.vivit.encoder.layer = self.model.vivit.encoder.layer[:hidden_layers]

    def forward(self, x):
        x = self.model(pixel_values=x)
        return x.logits
    
    def prepare_images(self, x):
        return self.image_processor(x, return_tensors="pt")
    
    def get_image_processor(self):
        return self.image_processor

    def save_weight(self, path:str):
        torch.save(self.model.state_dict(), path)

    def load_weight(self, path:str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
