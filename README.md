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
| **Model**| **Test Accuracy**     | **Inference Time (s)**  | **Test Accuracy**     | **Inference Time (s)**  |
| Default   | 77.9%  | 1.994 | 36.12%    | 2.003   |
| Post-Training 1.58-Bit Quantization on Linear Layers   | 67.81%  | 1.97  | 10.77%   | 1.971 |
| Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers   | 11.32%      | 1.913       | 1.68%      | 1.916    |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers   | 29.53%      | 1.952       | 6.9%       | 1.948   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers   | 66.2%      | 1.952       | 10.65%       | 1.962   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers   | 61.94%      | 1.945       | 8.5%       | 1.961   |


ResNet-18 Results

|          | **CIFAR-10**          |                     | **CIFAR-100**         |                     |
|:--------:|:---------------------:|:-------------------:|:---------------------:|:-------------------:|
| **Model**| **Test Accuracy**     | **Inference Time (s)**  | **Test Accuracy**     | **Inference Time (s)**  |
| Default   | 80.12%  | 2.583 | 50.64%    | 2.558   |
| Post-Training 1.58-Bit Quantization on Linear Layers   | 79.28%  | 2.568  | 44.67%  | 2.57 |
| Post-Training 1.58-Bit Quantization on Linear and Convolutional Layers   | 17.95%      | 2.55      | 1.83%     | 2.556    |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% Random Pruning on Conv Layers   | 51.97%     | 2.554    | 10.19%      | 2.565   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L1 Pruning on Conv Layers   | 78.21%     | 2.559       | 42.62%       | 2.58   |
| Post-Training 1.58-Bit Quantization on Linear Layers Followed with 30% L2 Pruning on Conv Layers   | 59.14%      |  2.584        | 13.58%      | 2.56   |



ViT Results

|          | **CIFAR-10**          |                     |
|:--------:|:---------------------:|:-------------------:|
| **Model**| **Test Accuracy**     | **Inference Time (s)**  | 
| Default   | 98.5%  | 121.91 | x    | x   |
| Post-Training 1.58-Bit Quantization on Linear Layers   | 10.04%  | 102.715  | 
| Quantization-Aware Training (1.58-Bit) on Linear Layers   | 67.78%  | 113.28     | 
| Quantization-Aware Training (1.58-Bit) on Linear Layer Followed with 30% Random Pruning on Conv Layers   | 53.8%      | 109.98       | 
| Quantization-Aware Training (1.58-Bit) on Linear Layer Followed with 30% L1 Pruning on Conv Layers   | 64.73%     | 111.59      |
| Quantization-Aware Training (1.58-Bit) on Linear Layer Followed with 30% L2 Pruning on Conv Layers   | 50.22%     | 110.42     | 