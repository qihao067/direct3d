import torch
import torch.nn as nn
from .modules import FrozenOpenCLIPEmbedder

class PromptModel(nn.Module):
    """main class"""

    def __init__(self):
        super().__init__()
        self.cond_stage_model = FrozenOpenCLIPEmbedder(layer='penultimate')
    
    def get_learned_conditioning(self, cond):
        cond = self.cond_stage_model.encode(cond)
        if isinstance(cond, DiagonalGaussianDistribution):
            cond = cond.mode()
        return cond