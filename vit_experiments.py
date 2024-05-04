import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import numpy as np
from typing import Type
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from models.vit import VITModelNotQuantized
from quantization_utils.bit_linear_custom import BitLinear

from quantization_utils.quantization_functions import QuantizationUtilityFunctions
from train_test_utils.train_test_functions import TrainTestUtils


import wandb
wandb.login()

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset_cifar_10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader_cifar_10 = torch.utils.data.DataLoader(trainset_cifar_10, batch_size=32,
                                          shuffle=True, num_workers=2)

testset_cifar_10 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader_cifar_10 = torch.utils.data.DataLoader(testset_cifar_10, batch_size=32,
                                         shuffle=False, num_workers=2)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset_cifar_100 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader_cifar_100 = torch.utils.data.DataLoader(trainset_cifar_100, batch_size=32,
                                          shuffle=True, num_workers=2)

testset_cifar_100 = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader_cifar_100 = torch.utils.data.DataLoader(testset_cifar_100, batch_size=32,
                                         shuffle=False, num_workers=2)

  

print("Training vit")

vit = VITModelNotQuantized.vit_model.to(device)
TrainTestUtils.train(device, vit, trainloader_cifar_10, "vit-CIFAR-10-NoQuantization", 5)
TrainTestUtils.test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-10-NoQuantization", model_to_test = vit, data_loader = testloader_cifar_10)

print("post quantization training for vit")
vit_quantized_linear = QuantizationUtilityFunctions.copy_model(vit)
QuantizationUtilityFunctions.quantize_layer_weights(device, vit_quantized_linear)
TrainTestUtils.test_and_export_logs(device, "vit-CIFAR-10-PostTrainingQuantizationLinear", vit_quantized_linear, testloader_cifar_10)

print("quantization aware training for vit")
vit_quantized_aware = VITModelNotQuantized.vit_model.to(device)
def replace_linear_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Set the new layer with the same parameters
            setattr(module, name, BitLinear(child.in_features, child.out_features, child.bias is not None))
        else:
            # Recursively apply the function to children
            replace_linear_layers(child)

# Apply the function to the entire model
replace_linear_layers(vit_quantized_aware)
vit_quantized_aware.to(device)
TrainTestUtils.train(device, vit, trainloader_cifar_10, "vit-CIFAR-10-QuantizationAwareLinear", 5)
TrainTestUtils.test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-10-QuantizationAwareLinear", model_to_test = vit_quantized_aware, data_loader = testloader_cifar_10)


