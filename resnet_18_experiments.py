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
from models.resnet_18_custom import BasicBlock
from models.resnet_18_custom import ResNet
from models.resnet_18_custom import ResNet
from models.resnet_18_custom import ResNetQuantized

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
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset_cifar_100 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader_cifar_100 = torch.utils.data.DataLoader(trainset_cifar_100, batch_size=32,
                                          shuffle=True, num_workers=2)

testset_cifar_100 = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader_cifar_100 = torch.utils.data.DataLoader(testset_cifar_100, batch_size=32,
                                         shuffle=False, num_workers=2)

  

print("Testing CIFAR-10")

resnet_18 = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
TrainTestUtils.train(device, resnet_18, trainloader_cifar_10, "ResNet18-CIFAR-10-NoQuantization", 1)
TrainTestUtils.test_and_export_logs(device = device, wandb_log_name = "ResNet18-CIFAR-10-NoQuantization", model_to_test = resnet_18, data_loader = testloader_cifar_10)

resnet_18_quantized_linear = QuantizationUtilityFunctions.copy_model(resnet_18)
QuantizationUtilityFunctions.quantize_layer_weights(device, resnet_18_quantized_linear)
TrainTestUtils.test_and_export_logs(device, "ResNet18-CIFAR-10-PostTrainingQuantizationLinear", resnet_18_quantized_linear, testloader_cifar_10)


resnet_18_quantized_linear_and_conv = QuantizationUtilityFunctions.copy_model(resnet_18)
QuantizationUtilityFunctions.quantize_layer_weights_including_conv(device, resnet_18_quantized_linear_and_conv)
TrainTestUtils.test_and_export_logs(device, "ResNet18-CIFAR-10-PostTrainingQuantizationLinearAndConv", resnet_18_quantized_linear_and_conv, testloader_cifar_10)


print("Now Testing CIFAR-100")

resnet_18 = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=100).to(device)
TrainTestUtils.train(device, resnet_18, trainloader_cifar_100, "ResNet18-CIFAR-100-NoQuantization", 1)
TrainTestUtils.test_and_export_logs(device, "ResNet18-CIFAR-100-NoQuantization", resnet_18, testloader_cifar_100)


resnet_18_quantized_linear = QuantizationUtilityFunctions.copy_model(resnet_18)
QuantizationUtilityFunctions.quantize_layer_weights(device, resnet_18_quantized_linear)
TrainTestUtils.test_and_export_logs(device, "ResNet18-CIFAR-100-PostTrainingQuantizationLinear", resnet_18_quantized_linear, testloader_cifar_100)


resnet_18_quantized_linear_and_conv = QuantizationUtilityFunctions.copy_model(resnet_18)
QuantizationUtilityFunctions.quantize_layer_weights_including_conv(device, resnet_18_quantized_linear_and_conv)
TrainTestUtils.test_and_export_logs(device, "ResNet18-CIFAR-100-PostTrainingQuantizationLinearAndConv", resnet_18_quantized_linear_and_conv, testloader_cifar_100)
