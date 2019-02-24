import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from random import choice

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
                f, t, Sxx = scipy.signal.spectrogram(data,rate,mode="angle")

                self.data[i] = (rate,Sxx)

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

