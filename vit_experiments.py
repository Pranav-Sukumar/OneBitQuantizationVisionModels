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
from models.vit import VITModelNotQuantized, VITModelNotQuantizedLarge
from quantization_utils.bit_linear_custom import BitLinear

from quantization_utils.quantization_functions import QuantizationUtilityFunctions
from pruning_utils.pruning_functions import PruningUtils

import wandb

NUM_EPOCHS = 7
wandb.login()

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

# Need to scale image from 32x32 to 224x224 for pretrained model
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

def train(device, model: nn.Module, dataloader: DataLoader, experiment_name, num_epochs = 5):
    wandb.init(
        # Set the project where this run will be logged
        project="OnePointFiveBitQuantizationResultsFinal",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=experiment_name,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 0.001,
        "architecture": "ResNet18",
        "epochs": num_epochs,
        })
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total_batch_loss = 0.0
        num_batches = 0

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            num_batches += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_batch_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        average_batch_loss = total_batch_loss / num_batches
        wandb.log({"loss": average_batch_loss})
        print(f"Epoch {epoch} has a loss of {average_batch_loss}")
    print('Finished Training')
    wandb.finish()

def test(device, model: nn.Module, dataloader: DataLoader, max_samples=None) -> float:
    correct = 0
    total = 0
    n_inferences = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)  # Access the 'logits' attribute instead of 'data'
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if max_samples:
                n_inferences += images.shape[0]
                if n_inferences > max_samples:
                    break

    return 100 * correct / total

def test_and_export_logs(device, wandb_log_name, model_to_test, data_loader):
        wandb.init(
        project="OnePointFiveBitQuantizationResultsFinal",
        name=wandb_log_name,
        )
        s = time.time()
        for i in range(5):
            score = test(device = device, model = model_to_test, dataloader = data_loader)
        average_inference_time = (time.time() - s) / 5
        print(average_inference_time)
        print('Accuracy of the network on the test images: {}%'.format(score))

        wandb.log({"Test Accuracy": score})
        wandb.log({"Average Inference Time": average_inference_time})
        
        wandb.finish()

'''
print("Training vit CIFAR-10")

vit = VITModelNotQuantized.vit_model.to(device)
train(device, vit, trainloader_cifar_10, "vit-CIFAR-10-NoQuantization", NUM_EPOCHS)
test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-10-NoQuantization", model_to_test = vit, data_loader = testloader_cifar_10)

torch.save(vit.state_dict(), "vit.pth")

print("post quantization training for vit")
vit_quantized_linear = QuantizationUtilityFunctions.copy_model(vit)
QuantizationUtilityFunctions.quantize_layer_weights(device, vit_quantized_linear)
test_and_export_logs(device, "vit-CIFAR-10-PostTrainingQuantizationLinear", vit_quantized_linear, testloader_cifar_10)

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


replace_linear_layers(vit_quantized_aware)
vit_quantized_aware.to(device)
train(device, vit, trainloader_cifar_10, "vit-CIFAR-10-QuantizationAwareLinear", NUM_EPOCHS)
test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-10-QuantizationAwareLinear", model_to_test = vit_quantized_aware, data_loader = testloader_cifar_10)

torch.save(vit.state_dict(), "vit_quantize_aware.pth")

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_l2_structured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-10-QuantizationAwarePrunedIterative", vit_quantized_aware_pruned_conv, testloader_cifar_10)

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_l1_unstructured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-10-QuantizationAwarePrunedL1Unstructured", vit_quantized_aware_pruned_conv, testloader_cifar_10)

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_random_unstructured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-10-QuantizationAwarePrunedRandomUnstructured", vit_quantized_aware_pruned_conv, testloader_cifar_10)

'''
print("Training vit CIFAR-100")


vit = VITModelNotQuantizedLarge.vit_model.to(device)
train(device, vit, trainloader_cifar_100, "vit-CIFAR-100-NoQuantization", NUM_EPOCHS)
test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-100-NoQuantization", model_to_test = vit, data_loader = testloader_cifar_100)

torch.save(vit.state_dict(), "vit.pth")

print("post quantization training for vit")
vit_quantized_linear = QuantizationUtilityFunctions.copy_model(vit)
QuantizationUtilityFunctions.quantize_layer_weights(device, vit_quantized_linear)
test_and_export_logs(device, "vit-CIFAR-100-PostTrainingQuantizationLinear", vit_quantized_linear, testloader_cifar_100)

print("quantization aware training for vit")
vit_quantized_aware = VITModelNotQuantizedLarge.vit_model.to(device)
def replace_linear_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Set the new layer with the same parameters
            setattr(module, name, BitLinear(child.in_features, child.out_features, child.bias is not None))
        else:
            # Recursively apply the function to children
            replace_linear_layers(child)


replace_linear_layers(vit_quantized_aware)
vit_quantized_aware.to(device)
train(device, vit, trainloader_cifar_10, "vit-CIFAR-100-QuantizationAwareLinear", NUM_EPOCHS)
test_and_export_logs(device = device, wandb_log_name = "vit-CIFAR-100-QuantizationAwareLinear", model_to_test = vit_quantized_aware, data_loader = testloader_cifar_100)

torch.save(vit.state_dict(), "vit_quantize_aware.pth")

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_l2_structured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-100-QuantizationAwarePrunedIterative", vit_quantized_aware_pruned_conv, testloader_cifar_100)

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_l1_unstructured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-100-QuantizationAwarePrunedL1Unstructured", vit_quantized_aware_pruned_conv, testloader_cifar_100)

vit_quantized_aware_pruned_conv = QuantizationUtilityFunctions.copy_model(vit_quantized_aware)
PruningUtils.prune_model_random_unstructured(vit_quantized_aware_pruned_conv)
test_and_export_logs(device, "vit-CIFAR-100-QuantizationAwarePrunedRandomUnstructured", vit_quantized_aware_pruned_conv, testloader_cifar_100)