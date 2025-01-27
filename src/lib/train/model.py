from transformers import TimesformerForVideoClassification, TimesformerConfig, AutoImageProcessor
import torch

class MyModel(torch.nn.Module):
    def __init__(self, base_model:str = "facebook/timesformer-base-finetuned-k400", *args, **kwargs):
        super().__init__(*args, **kwargs)
        configuration = TimesformerConfig()
        # "facebook/timesformer-base-finetuned-k400"
        self.model = TimesformerForVideoClassification.from_pretrained(base_model)
        num_classes = 1
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes, bias=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            base_model
        )

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
