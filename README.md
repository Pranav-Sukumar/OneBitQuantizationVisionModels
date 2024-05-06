# OneBitQuantizationVisionModels
## HPML Project by Pranav Sukumar (ps3388)

### Project Description
#### A. Background and Motivation
The increasing popularity of large transformer models has led to breakthroughs in model performance but at the cost of blowing up compute requirements. Every year, model size increases by 10 times. Memory is expensive and requires large amounts of energy to move at scale. Additionally, the size of these large models makes it difficult to run inference on edge devices with lower computing capabilities. 

Model Quantization and Pruning are two strategies that can be used to decrease the memory requirements and inference time of a deep neural network. In this analysis, I examine the effects of 1.58-bit quantization coupled with pruning. 1-bit and 1.58-bit quantization are very new quantization ideas that have been applied to NLP tasks. This analysis focuses on the effects of 1.58-bit quantization coupled with pruning for a different domain: vision models.

#### B. Problem Statement
The goal of this research is to analyze the effects of 1.58-bit quantization coupled with pruning for various image classification models. Specifically, I analyze how detrimental these techniques are for model accuracy, and what benefits they bring for model size and inference time.

#### C. Objectives and Scope
The primary objective of this research is to evaluate the impact of 1.58-bit quantization combined with pruning on the performance of vision models used for object classification. This study aims to:
1. Assess Model Accuracy: Quantify the impact of 1.58-bit quantization and pruning on the accuracy of various vision models.
2. Analyze Model Size Reduction: Measure the reduction in model size resulting from the implementation of 1.58-bit quantization and pruning.
3. Evaluate Inference Efficiency: Investigate the improvements in inference time when models are quantized and pruned.
4. Provide Implementation Guidelines: Offer insights and recommendations on implementing these techniques effectively

### An Outline of the Repository
The three models I chose for analysis are AlexNet, ResNet-18, and a Vision Transformer. In the codebase, these 3 models are all defined in Pytorch in the models module. The quantization_utils module contains two python files, one for post training quantization functions and the other for the bit_linear layer for quantization-aware training for the ViT. The pruning_utils module contains one python file which has functions related to pruning. Since the training and testing loops were largely the same for the experiments, I put the logic into a separate module called train_test_utils. Splitting up the logic into all these modules makes the code more organized and easy to read.

In the main folder of the repository there are 3 python files for the experiments pertaining to each model (AlexNet, ResNet-18, and ViT). Each of these files trains the model, does 1.58-bit quantization, and tries different pruning strategies. Each experiment does this for both the CIFAR-10 and CIFAR-100 datasets. There is one additional python file determine_space_saved.py which does some calculations for the total space saved with quantization. Each experiment uses Weights and Biases to track and export logs. 

### Commands to Execute the Code
I created a requirements.txt file which can be used to install all the libraries needed for this repository. However, the requirements.txt is likely overkill and the only libraries that need to be installed should be "torch", "torchvision", "transformers", and "wandb". After doing this, you can run any of the experiments by simply calling "python3 alexnet_experiments.py", "python3 resnet_18_experiments.py", "python3 vit_experiments.py", or "python3 determine_space_saved.py". I ran the code in the GCP VM that had a T4 GPU. The code has been seeded so you should expect to get the same results as those shown below and in the report.

### Results
For a more detailed breakdown of the results, please see the written report. Here is a summary of the results:

AlexNet Results

|          | **CIFAR-10**          |                     | **CIFAR-100**         |                     |
|:--------:|:---------------------:|:-------------------:|:---------------------:|:-------------------:|
| **Model**| **Test Accuracy**     | **Inference Time**  | **Test Accuracy**     | **Inference Time**  |
| Default   | 77.9%  | 1.994 | 36.12%    | 2.003   |
| Post-Training 1.58-Bit Quantization on Linear Layers   | 67.81%  | 1.97  | 10.77%   | 1.971 |
| Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers   | 11.32%      | 1.913       | 1.68%      | 1.916    |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers   | 29.53%      | 1.952       | 6.9%       | 1.948   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers   | 66.2%      | 1.952       | 10.65%       | 1.962   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers   | 61.94%      | 1.945       | 8.5%       | 1.961   |


CIFAR-10 AlexNet:
AlexNet Default Test Accuracy (no quantization): 77.9%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Test Accuracy: 67.81%
AlexNet Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Test Accuracy: 11.32%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Test Accuracy: 29.53%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Test Accuracy: 66.2%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Test Accuracy: 61.94%

CIFAR-100 AlexNet:
AlexNet Default Test Accuracy (no quantization): 36.12%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Test Accuracy: 10.77%
AlexNet Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Test Accuracy: 1.68%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Test Accuracy: 6.9%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Test Accuracy: 10.65%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Test Accuracy: 8.5%

CIFAR-10 AlexNet:
AlexNet Default Inference Time (no quantization): 77.9%
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Inference Time:

CIFAR-100 AlexNet:
AlexNet Default Inference Time (no quantization):
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Inference Time:
AlexNet Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Inference Time:


CIFAR-10 ResNet-18:
ResNet-18 Default Test Accuracy (no quantization): 80.12%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Test Accuracy: 79.28%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Test Accuracy: 17.95%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Test Accuracy: 51.97%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Test Accuracy: 78.21%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Test Accuracy: 59.14%

CIFAR-100 ResNet-18:
ResNet-18 Default Test Accuracy (no quantization): 50.64%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Test Accuracy: 44.67%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Test Accuracy: 1.83%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Test Accuracy: 10.19%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Test Accuracy: 42.62%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Test Accuracy: 13.58%


CIFAR-10 ResNet-18:
ResNet-18 Default Inference Time (no quantization): 77.9%
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Inference Time:

CIFAR-100 ResNet-18:
ResNet-18 Default Inference Time (no quantization):
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers Inference Time:
ResNet-18 Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers Inference Time: