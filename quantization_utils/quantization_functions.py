from copy import deepcopy
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import Tensor
from typing import Type
from typing import Tuple, List
import numpy as np


class QuantizationUtilityFunctions:

    def copy_model(model: nn.Module) -> nn.Module:
        result = deepcopy(model)

        # Copy over the extra metadata we've collected which copy.deepcopy doesn't capture
        if hasattr(model, 'input_activations'):
            result.input_activations = deepcopy(model.input_activations)

        num_layers_quantized = 0
        total_layers = 0

        def recursive_layer_copy(result_layer, original_layer):
            nonlocal num_layers_quantized, total_layers
            if isinstance(original_layer, nn.Conv2d) or isinstance(original_layer, nn.Linear):
                num_layers_quantized += 1
                total_layers += 1

                if hasattr(original_layer.weight, 'scale'):
                    result_layer.weight.scale = deepcopy(original_layer.weight.scale)
                if hasattr(original_layer, 'activations'):
                    result_layer.activations = deepcopy(original_layer.activations)
                if hasattr(original_layer, 'output_scale'):
                    result_layer.output_scale = deepcopy(original_layer.output_scale)

            else:
                # Handle other possible nested modules such as nn.Sequential, etc.
                for child_result_layer, child_original_layer in zip(result_layer.children(), original_layer.children()):
                    recursive_layer_copy(child_result_layer, child_original_layer)


        # Initialize the recursive copy process
        recursive_layer_copy(result, model)

        #print(f"Total layers: {total_layers}")
        #print(f"Quantized layers: {num_layers_quantized}")

        return result

    def quantized_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        '''
        Quantize the weights so that all values are integers between -128 and 127.
        You may want to use the total range, 3-sigma range, or some other range when
        deciding just what factors to scale the float32 values by.

        Parameters:
        weights (Tensor): The unquantized weights

        Returns:
        (Tensor, float): A tuple with the following elements:
                            * The weights in quantized form, where every value is an integer between -128 and 127.
                            The "dtype" will still be "float", but the values themselves should all be integers.
                            * The scaling factor that your weights were multiplied by.
                            This value does not need to be an 8-bit integer.
        '''


        weights = torch.clamp(weights, min=-128, max=127)
        weights = weights - torch.mean(weights)
        q_max = 3
        q_min = 1
        r_max = torch.max(weights).item()
        r_min = torch.min(weights).item()

        scale = (r_max - r_min) / (q_max - q_min)

        # Note that since I am doing r_max - r_min as the numerator, to calculate the quantized values, you must divide by scale, not multiply
        result = (weights  / scale).round()

        return torch.clamp(result, min=-1, max=1), scale

    def quantize_layer_weights(device, model):
        layers_quantized = 0
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                layers_quantized += 1
                q_layer_data, scale = QuantizationUtilityFunctions.quantized_weights(layer.weight.data)
                q_layer_data = q_layer_data.to(device)

                layer.weight.data = q_layer_data
                layer.weight.scale = scale

                if (q_layer_data < -128).any() or (q_layer_data > 127).any():
                    raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
                if (q_layer_data != q_layer_data.round()).any():
                    raise Exception("Quantized weights of {} layer include non-integer values".format(layer.__class__.__name__))
            else:
                QuantizationUtilityFunctions.quantize_layer_weights(device, layer)

        #print(f"Quantized layers: {layers_quantized}")

    def quantize_layer_weights_including_conv(device, model):
        layers_quantized = 0
        for layer in model.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                layers_quantized  += 1
                q_layer_data, scale = QuantizationUtilityFunctions.quantized_weights(layer.weight.data)
                q_layer_data = q_layer_data.to(device)

                layer.weight.data = q_layer_data
                layer.weight.scale = scale

                if (q_layer_data < -128).any() or (q_layer_data > 127).any():
                    raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
                if (q_layer_data != q_layer_data.round()).any():
                    raise Exception("Quantized weights of {} layer include non-integer values".format(layer.__class__.__name__))
            else:
                QuantizationUtilityFunctions.quantize_layer_weights_including_conv(device, layer)
        #print(f"Quantized layers: {layers_quantized}")
        

