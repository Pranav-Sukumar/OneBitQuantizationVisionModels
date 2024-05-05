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
from models.alex_net_custom import AlexNet
from quantization_utils.quantization_functions import QuantizationUtilityFunctions
from train_test_utils.train_test_functions import TrainTestUtils
from pruning_utils.pruning_functions import PruningUtils

import wandb
wandb.login()

NUM_EPOCHS = 30

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

alexnet = AlexNet(num_classes=10).to(device)
TrainTestUtils.train(device, alexnet, trainloader_cifar_10, "AlexNet-CIFAR-10-NoQuantization", NUM_EPOCHS)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-NoQuantization", alexnet, testloader_cifar_10)


alexnet_quantized_linear = QuantizationUtilityFunctions.copy_model(alexnet)
QuantizationUtilityFunctions.quantize_layer_weights(device, alexnet_quantized_linear)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-PostTrainingQuantizationLinear", alexnet_quantized_linear, testloader_cifar_10)


alexnet_quantized_linear_and_conv = QuantizationUtilityFunctions.copy_model(alexnet)
QuantizationUtilityFunctions.quantize_layer_weights_including_conv(device, alexnet_quantized_linear_and_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-PostTrainingQuantizationLinearAndConv", alexnet_quantized_linear_and_conv, testloader_cifar_10)


alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_l2_structured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-PostTrainingQuantizationLinearPrunedIterative", alexnet_quantized_linear_pruned_conv, testloader_cifar_10)

alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_l1_unstructured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-PostTrainingQuantizationLinearPrunedL1Unstructured", alexnet_quantized_linear_pruned_conv, testloader_cifar_10)

alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_random_unstructured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-10-PostTrainingQuantizationLinearPrunedRandomUnstructured", alexnet_quantized_linear_pruned_conv, testloader_cifar_10)

print("Now Testing CIFAR-100")

alexnet = AlexNet(num_classes=100).to(device)
TrainTestUtils.train(device, alexnet, trainloader_cifar_100, "AlexNet-CIFAR-100-NoQuantization", NUM_EPOCHS)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-NoQuantization", alexnet, testloader_cifar_100)


alexnet_quantized_linear = QuantizationUtilityFunctions.copy_model(alexnet)
QuantizationUtilityFunctions.quantize_layer_weights(device, alexnet_quantized_linear)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-PostTrainingQuantizationLinear", alexnet_quantized_linear, testloader_cifar_100)


alexnet_quantized_linear_and_conv = QuantizationUtilityFunctions.copy_model(alexnet)
QuantizationUtilityFunctions.quantize_layer_weights_including_conv(device, alexnet_quantized_linear_and_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-PostTrainingQuantizationLinearAndConv", alexnet_quantized_linear_and_conv, testloader_cifar_100)


alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_l2_structured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-PostTrainingQuantizationLinearPrunedIterative", alexnet_quantized_linear_pruned_conv, testloader_cifar_100)

alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_l1_unstructured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-PostTrainingQuantizationLinearPrunedL1Unstructured", alexnet_quantized_linear_pruned_conv, testloader_cifar_100)

alexnet_quantized_linear_pruned_conv = QuantizationUtilityFunctions.copy_model(alexnet_quantized_linear)
PruningUtils.prune_model_random_unstructured(alexnet_quantized_linear_pruned_conv)
TrainTestUtils.test_and_export_logs(device, "AlexNet-CIFAR-100-PostTrainingQuantizationLinearPrunedRandomUnstructured", alexnet_quantized_linear_pruned_conv, testloader_cifar_100)