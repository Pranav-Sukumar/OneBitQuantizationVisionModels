import torch
import torch.nn.utils.prune as prune

class PruningUtils:
    def prune_model_l1_unstructured(model):
        for name, module in model.named_modules():
            # prune 30% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
                
    def prune_model_random_unstructured(model):
        for name, module in model.named_modules():
            # prune 30% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.random_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
                
    def prune_model_iterative(model):
        for name, module in model.named_modules():
            # prune 30% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
                prune.remove(module, 'weight')
        
