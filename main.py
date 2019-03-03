import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

from args import parse_arguments
import util
from model import SVSGan
import train

if __name__ == '__main__':
    args = parse_arguments()

    batch_size = args.batch_size;
    # Continue placing other arguments here

    data_loader = util.get_loader(args)

    model = SVSGan(N_FFT,lrG,lrD,Gbeta1,Gbeta2,Dbeta1,Dbeta2,batch_size, hidden_nodes_G,hidden_nodes_D)

    for i in range(args.steps):
        for j in data_loader: 
            print(j)