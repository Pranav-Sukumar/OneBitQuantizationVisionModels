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


def train(model: nn.Module, dataloader: DataLoader, experiment_name, num_epochs = 5):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(
      # Set the project where this run will be logged
      project="OneBitQuantization",
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
            loss = criterion(outputs, labels)
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

def test(model: nn.Module, dataloader: DataLoader, max_samples=None) -> float:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    correct = 0
    total = 0
    n_inferences = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if max_samples:
                n_inferences += images.shape[0]
                if n_inferences > max_samples:
                    break

    return 100 * correct / total


resnet_18 = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)

train(resnet_18, trainloader_cifar_10, "ResNet18-CIFAR-10-NoQuantization", 1)

wandb.init(
  # Set the project where this run will be logged
  project="OneBitQuantization",
  # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
  name="ResNet18-CIFAR-10-NoQuantization",
  # Track hyperparameters and run metadata
)
s = time.time()
for i in range(30):
  score = test(resnet_18, testloader_cifar_10)
average_inference_time = (time.time() - s) / 30
print(average_inference_time)
print('Accuracy of the network on the test images: {}%'.format(score))

wandb.log({"Test Accuracy": score})
wandb.log({"Average Inference Time": average_inference_time})

wandb.finish()





wandb.init(
  # Set the project where this run will be logged
  project="OneBitQuantization",
  # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
  name="ResNet18-CIFAR-10-PostTrainingQuantizationLinear",
  # Track hyperparameters and run metadata
)

resnet_18_quantized_linear = QuantizationUtilityFunctions.copy_model(resnet_18)
QuantizationUtilityFunctions.quantize_layer_weights(device, resnet_18_quantized_linear)

s = time.time()
for i in range(30):
  score = test(resnet_18_quantized_linear, testloader_cifar_10)
average_inference_time = (time.time() - s) / 30
print(average_inference_time)
print('Accuracy of the network after quantizing all linear weights on CIFAR-10: {}%'.format(score))

#print(f"Size of model is {print_model_size(resnet_18_quantized_linear)}")

wandb.log({"Test Accuracy": score})
wandb.log({"Average Inference Time": average_inference_time})

wandb.finish()
