from models.vit import VITModelNotQuantized
from models.alex_net_custom import AlexNet
from models.resnet_18_custom import ResNet
from models.resnet_18_custom import BasicBlock

import torch
import torchvision
import torch.nn as nn
import os

# Function to count the number of linear weights
def count_linear_weights(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += module.weight.numel()  # Count weights
    return count

def calculate_original_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    size_bytes = os.path.getsize("tmp.pt")
    #print(f"{size_bytes} Bytes")
    os.remove('tmp.pt')
    return size_bytes

def calculate_quantized_model_size(model):
    num_linear_weights = count_linear_weights(model)
    space_saved_per_weight_bits = 32-3
    
    space_saved_total_bits = num_linear_weights * space_saved_per_weight_bits
    
    new_space_bytes = calculate_original_model_size(model) - (space_saved_total_bits/8)
    return new_space_bytes

alexnet = AlexNet(num_classes=10)
print(f"Alex Net Original Model Size: {calculate_original_model_size(alexnet)} Bytes")
print(f"Alex Net Quantized Model Ideal Size: {calculate_quantized_model_size(alexnet)} Bytes")


resnet = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10)
print(f"ResNet-18 Original Model Size: {calculate_original_model_size(resnet)} Bytes")
print(f"ResNet-18 Quantized Model Ideal Size: {calculate_quantized_model_size(resnet)} Bytes")
    
vit = VITModelNotQuantized.vit_model
print(f"ViT Original Model Size: {calculate_original_model_size(vit)} Bytes")
print(f"ViT Quantized Model Ideal Size: {calculate_quantized_model_size(vit)} Bytes")