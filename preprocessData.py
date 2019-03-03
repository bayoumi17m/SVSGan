import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

from args import parse_arguments_Data
import util

if __name__ == "__main__":
    args = parse_arguments_Data()
    util.DataSetCleaner(args)