# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        if isinstance(original_layer, nn.Conv2d):
            in_channels = original_layer.in_channels
            out_channels = original_layer.out_channels
            self.lora_A = nn.Parameter(torch.randn(rank, in_channels, 1, 1) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1, 1))
            self.scale = 0.1
            
        # Ensure LoRA parameters require gradients
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        
        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_output = self.original_layer(x)
        
        if isinstance(self.original_layer, nn.Conv2d):
            lora_output = F.conv2d(x, self.lora_A)
            lora_output = F.conv2d(lora_output, self.lora_B)
            return original_output + (self.scale * lora_output)
        
        return original_output

def load_model(model_path='models/vgg16_lora.pth'):
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Apply LoRA to all Conv2d layers in the VGG16 model
    for i, layer in enumerate(vgg16.features):
        if isinstance(layer, nn.Conv2d):
            vgg16.features[i] = LoRALayer(layer, rank=4)
    
    # Freeze all layers except LoRA layers
    for name, param in vgg16.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    # Unfreeze the new classifier layers
    for param in vgg16.classifier[6].parameters():
        param.requires_grad = True

    # Load pretrained weights if available
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        vgg16.load_state_dict(state_dict, strict=False)
    
    return vgg16


def predict_image(model, image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    return predicted.item()