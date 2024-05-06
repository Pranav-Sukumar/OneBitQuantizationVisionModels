import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import ViTForImageClassification
import torch

class VITModelNotQuantized:
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 10
    
    vit_model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    
class VITModelNotQuantizedLarge:
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 100
    
    vit_model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=num_labels)




