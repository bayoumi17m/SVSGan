import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

def evaluate(model,in_mixture,separated_source=None):
    """Docstring for evaluate"""

    predicted_source = model.gen(in_mixture)
    if separated_source is None:
        return predicted_source, None
    else:
        return predicted_source, model.dis(predicted_source)