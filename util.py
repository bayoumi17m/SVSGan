import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DSD100Dataset(Dataset):
    """DOcstring for DSD100 Dataset"""

    def __init__(self, root_dir, length):
        """Docstring for the Dataset object"""
