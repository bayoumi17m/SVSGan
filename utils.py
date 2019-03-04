import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from random import choice
import numpy as np
import scipy
from scipy import signal
import scipy.io.wavfile
from scipy.io.wavfile import read
from pydub import AudioSegment
import fnmatch

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='mnist', help='Load a previous dataset')
    parser.add_argument('--train_directory', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--val_directory', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--test_directory', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--log-dir', type=str, default=None, help='Logging directory (default None)')
    parser.add_argument('--store_data', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='File to resume')
    parser.add_argument('--resume_G', type=str, default=None, help='File to resume')
    parser.add_argument('--resume_D', type=str, default=None, help='File to resume')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train it.')
    parser.add_argument('--log-step', type=int, default=10, help='Logging step to the terminal.')
    parser.add_argument('--save-step', type=int, default=1, help='Number of steps to save it.')
    parser.add_argument('--val-freq', type=int, default=1, help='Validation frequency (unit: epochs).')
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
    parser.add_argument('--train', action="store_true", default=True, help='Training mode')
    args = parser.parse_args()

    return args


def DataSetCleaner(dataroot,store_data):
    for filename in os.listdir(dataroot):
        if filename.endswith(".wav"):
            try:
                os.mkdir(os.path.join(store_data,filename[:-4]))
            except:
                pass
            rate, data = scipy.io.wavfile.read(os.path.join(dataroot,filename))
            f, t, Sxx = signal.stft(data,rate,nperseg=1000)
            magnitude = np.abs(Sxx)
            phase = np.unwrap(np.angle(Sxx),axis=-2)
            np.save(os.path.join(os.path.join(store_data,filename[:-4]),"rate_"+ filename[:-4]),rate)
            np.save(os.path.join(os.path.join(store_data,filename[:-4]),"freq_"+ filename[:-4]),f)
            np.save(os.path.join(os.path.join(store_data,filename[:-4]),"time_"+ filename[:-4]),t)
            np.save(os.path.join(os.path.join(store_data,filename[:-4]),"magnitude_"+ filename[:-4]),magnitude)
            np.save(os.path.join(os.path.join(store_data,filename[:-4]),"phase_"+ filename[:-4]),phase)

def reConstructSound(filename,magnitude,phase,fs):
    Zxx = magnitude * np.exp(1j * phase)
    t2, xrec = signal.istft(Zxx, fs)
    scipy.io.wavfile.write(filename,fs,xrec)


class DSD100Dataset(Dataset):
    """DOcstring for DSD100 Dataset"""

    def __init__(self, root_dir):
        """Docstring for the Dataset object"""
        super(DSD100Dataset, self).__init__()
        #length = len(list(filter(lambda x: x[-3:] == ".npy", os.listdir("."))))
        subdirs = [x[0] for x in os.walk(root_dir)]
        length = len(subdirs)

        self.data = []

        print(root_dir)
        for song in os.listdir(root_dir):
            if song.startswith("."):
                continue
            data = ({},{},{})
            print(song)
            for song_portion in os.listdir(os.path.join(root_dir, song)):
                if song_portion.startswith("."):
                    continue
                k = -1
                if fnmatch.fnmatch(song_portion,"mixture"):
                    k = 0
                elif fnmatch.fnmatch(song_portion,"vocal"):
                    k = 1
                elif fnmatch.fnmatch(song_portion, "noise"):
                    k = 2
                print(k)

                for filename in os.listdir(os.path.join(root_dir, song, song_portion)):
                    print(filename)
                    print(fnmatch.filter(filename,"magnitude_*"))
                    prefix = filename.split("_")[0]
                    print(prefix)
                    data[k][prefix] = np.load(os.path.join(root_dir, song, song_portion, filename))
                print({key:v.shape for key,v in data[k].items()})

            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO : subsampling
        data = (self.data[idx][0], self.data[idx][1], self.data[idx][2])
        prefixIdx = ("mixture", "vocal", "noise")
        length = len(data)
        sIdx = np.random.randint(0,length- 1 - 10); eIdx = sIdx + 10
        for i in range(len(prefixIdx)):
            data[prefixIdx[i]] = {"magnitude": data[prefixIdx[i]]["magnitude"][:, sIdx:eIdx], "phase": data[prefixIdx[i]]["phase"][:, sIdx:eIdx]}
            

        #import pdb; pdb.set_trace()
        return data


def get_loader(args):
    train_dataset = DSD100Dataset(args.train_directory)
    val_dataset = DSD100Dataset(args.val_directory)
    test_dataset = DSD100Dataset(args.test_directory)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size= args.batch_size,
        shuffle=False, num_workers=args.workers)

    validation_data_loader = torch.utils.data.DataLoader(
        dataset = val_dataset, batch_size= args.batch_size,
        shuffle=False, num_workers=args.workers)

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size= args.batch_size,
        shuffle=False, num_workers=args.workers)

    return {"train":train_data_loader, "test": test_data_loader, "val": validation_data_loader}


dataset_paths = {
    'mixtures': os.path.join('DSD100DatasetPath', 'Mixtures'),
    'sources': os.path.join('DSD100DatasetPath', 'Sources')
}


def get_DSD_files(subset):
    """Getting the files lists.

    :param subset: The subset that we are interested in (i.e. training or testing).
    :type subset: str
    :return: The lists with the file paths of the files that we want to use.
    :rtype: (list[str], list[str])
    """
    specific_dir = 'Dev' if subset == 'training' else 'Test'
    mixtures_dir = os.path.join(dataset_paths['mixtures'], specific_dir)
    sources_dir = os.path.join(dataset_paths['sources'], specific_dir)

    mixtures_list = [os.path.join(mixtures_dir, file_path)
                     for file_path in sorted(os.listdir(mixtures_dir))]

    sources_list = [os.path.join(sources_dir, file_path)
                    for file_path in sorted(os.listdir(sources_dir))]

    return mixtures_list, sources_list

def DSD2np(files):
    mix = np.asarray([np.array(read(os.path.join(file, "mixture.wav"))[1], dtype=float)  for file in files[0]])
    vocals = np.asarray([np.array(read(os.path.join(file, "vocals.wav"))[1], dtype=float)  for file in files[1]])
    bgms = []
    for file in files[1]:
        bass = AudioSegment.from_wav(os.path.join(file, "bass.wav"))
        drums = AudioSegment.from_wav(os.path.join(file, "drums.wav"))
        other = AudioSegment.from_wav(os.path.join(file, "other.wav"))
        bgm = bass.overlay(drums.overlay(other))
        bgms.append(np.array(bgm.get_array_of_samples(), dtype=float))
    bgms = np.asarray(bgms)
    return mix, vocals, bgms


def get_train_test():
    mix_train, vocals_trian, bgm_train = DSD2np(get_DSD_files('training'))
    mix_test, vocals_test, bgm_test = DSD2np(get_DSD_files('testing'))
    return mix_train, vocals_trian, bgm_train, mix_test, vocals_test, bgm_test


