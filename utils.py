import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from random import choice
import numpy as np
import scipy

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='mnist', help='Load a previous dataset')
    parser.add_argument('--dataroot', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='File to resume')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--lrG', type=float, default=0.001,
            help='learning rate for generator, default=0.001')
    parser.add_argument('--lrD', type=float, default=0.001,
            help='learning rate for Discriminator, default=0.001')
    parser.add_argument('--Gbeta1', type=float, default=0.5, help='Generator beta1 for adam. default=0.5')
    parser.add_argument('--Gbeta2', type=float, default=0.999, help='Generator beta2 for adam. default=0.999')
    parser.add_argument('--Dbeta1', type=float, default=0.5, help='Discriminator beta1 for adam. default=0.5')
    parser.add_argument('--Dbeta2', type=float, default=0.999, help='Discriminator beta2 for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--seed', default=100, type=int, help='Random seed.')
    parser.add_argument('--load', action="store_true", help='load dataset')
    parser.add_argument('--ngf', type=int, default=1024, help='number of features in generator')
    parser.add_argument('--ndf', type=int, default=513, help='number of features in discriminator')
    parser.add_argument('--N_FFT', type=int, default=513, help='size of the input spectra of the generator')
    parser.add_argument('--inD', type=int, default=1539, help='size of the input features of the discriminator')
    args = parser.parse_args()

    return args

class DSD100Dataset(Dataset):
    """DOcstring for DSD100 Dataset"""

    def __init__(self, root_dir):
        """Docstring for the Dataset object"""
        length = len(list(filter(lambda x: x[-3:] == ".wav", os.listdir("."))))
        self.data = np.empty(length)
        i = 0
        for filename in os.listdir(root_dir):
            if filename.endswith(".wav"):
                rate, data = scipy.io.wavefile.read(filename)
                f, t, Sxx = scipy.signal.spectrogram(data,rate,mode="magnitude")
                fa, ta, Sxxa = scipy.signal.spectrogram(data,rate,mode="angle")

                self.data[i] = ((f,t,fa,ta,Sxxa),(rate,Sxx))

            else:
                pass

        def __len__(self):
            return self.data.shape[0]


        def __getitem__(self, _):
            return np.random.choice(self.data)


def get_loader(args):
    dataset = DSD100Dataset(args.dataroot)
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset, batch_size= args.batch_size, shuffle=True,num_workers=args.workers)

    return data_loader

