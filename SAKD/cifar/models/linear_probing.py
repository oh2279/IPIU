import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)