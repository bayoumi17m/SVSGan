import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

import utils

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='mnist', help='Load a previous dataset')
    parser.add_argument('--dataroot', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--store_data', type=str, default='./data/', help='path to dataset')

if __name__ == "__main__":
    args = parse_args()
    utils.DataSetCleaner(args.dataroot, args.store_data)