from composer.algorithms.layer_freezing.layer_freezing import _remove_param_from_optimizers
import torch
import collections
from composer.core import Callback, State
from composer.loggers import Logger
from typing import List

__all__ = ['ResumptionCallback', 'RESUMPTION_STRATEGIES', 'GlobalLRLowering']


class ResumptionCallback(Callback):
    def __init__(self,):
          pass
       
    def fit_start(self, state: State, logger: Logger):
        raise NotImplementedError


class GlobalLRLowering(ResumptionCallback):
    def __init__(self, lr_scale: float):
        self.lr_scale = lr_scale
       
    def fit_start(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        
        lrs = []
        new_lrs = []
        for optimizer in state.optimizers:
            for group in optimizer.param_groups:
                lrs.append(group['lr'])
                group['initial_lr'] = group['initial_lr'] * self.lr_scale
                group['lr'] = group['lr'] * self.lr_scale
                new_lrs.append(group('lr'))
        print(f"Lowers LRs from: {lrs} to {new_lrs}")

class LayerFreezing(ResumptionCallback):
    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
       
    def fit_start(self, state: State, logger: Logger):
        model_layers = set(name for name, _ in state.model.named_parameters())
        for layer in self.layer_names:
            if layer not in model_layers:
                raise Exception(f"Attempted to freeze layer not found in model: {layer}\nAvailable layers: {model_layers}")

        count = 0
        for name, p in state.model.named_parameters():
            if p.requires_grad and name in self.layer_names:
                p.requires_grad = False
                _remove_param_from_optimizers(p, state.optimizers)
                count += 1
        
        if count == 0:
            raise Exception(f"Tried to run LayerFreezing but didn't freeze any layers {count}")

RESUMPTION_STRATEGIES = {
    "layer_freezing": LayerFreezing,
    'global_lr_lowering': GlobalLRLowering
}
