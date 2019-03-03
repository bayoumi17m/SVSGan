import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

import utils
from model import SVSGan
import train

if __name__ == '__main__':
    args = utils.get_args()

    batch_size = args.batch_size;
    # Continue placing other arguments here

    data_loader = utils.get_loader(args)

    model = SVSGan(args)

    for i in range(args.steps):
        for j in data_loader: 
            print(j)