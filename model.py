import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os

class Generator(nn.Module):
    """Docstring for Generator"""
    def __init__(self,N_FFT,hidden_nodes_G):
        """Docstring for Init of Generator"""
        # We take in a vector of size 1024?
        self.fc1 = nn.Linear(N_FFT,hidden_nodes_G)
        self.fc2 = nn.Linear(hidden_nodes_G,hidden_nodes_G)
        self.fc3 = nn.Linear(hidden_nodes_G,hidden_nodes_G)
        self.fc4 = nn.Linear(hidden_nodes_G,N_FFT)

        self.bn1 = nn.BatchNorm(batch_size)
        self.bn2 = nn.BatchNorm(batch_size)
        self.bn3 = nn.BatchNorm(batch_size)
        self.bn4 = nn.BatchNorm(batch_size)


    def forward(self,z):
        """Docstring for forward function"""
        #z1 = 1 - z
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        # x1 = F.sigmoid(self.fc1(z1))
        # x1 = F.sigmoid(self.fc2(x1))
        # x1 = F.sigmoid(self.fc3(x1))
        # x1 = F.sigmoid(self.fc4(x1))

        x1 = 1 - x # Take probabilities and subtract out the probabilities for human voice

        return (x / (x + x1 + np.finfo(float).eps)) * z # Time Masking Function



    def save(self):
        """Docstring for saving"""


    def load(self):
            """Docstring for loading"""


class Discriminator(nn.Module):
    """Docstring for Discriminator"""

    def __init__(self,in_size,hidden_nodes_D,batch_size):
        """Docstring for Init of Discriminator"""
        self.fc1 = Linear(in_size,hidden_nodes_D)
        self.fc2 = Linear(hidden_nodes_D,hidden_nodes_D)
        self.fc3 = Linear(hidden_nodes_D,hidden_nodes_D)
        self.fc4 = Linear(hidden_nodes_D,1)
        
        self.bn1 = nn.BatchNorm(batch_size)
        self.bn2 = nn.BatchNorm(batch_size)
        self.bn3 = nn.BatchNorm(batch_size)
        self.bn4 = nn.BatchNorm(batch_size)



    def forward(self,z):
        """Docstring for forward function"""
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.bn4(self.fc4(x)))
        return x



    def save(self):
        """Docstring for saving"""


    def load(self):
            """Docstring for loading"""


class SVSGan(object):
    """Docstring for SVSGan"""

    def __init__(self,N_FFT,lrG,lrD,Gbeta1,Gbeta2,Dbeta1,Dbeta2,batch_size, hidden_nodes_G,hidden_nodes_D):
        """Docstring for init of SVSGan"""
        self.gen = Generator(N_FFT)
        self.dis = Discriminator(N_FFT)
        self.gen_optim = optim.Adam(self.gen.parameters(),lr=lrG,betas=(Gbeta1,Gbeta2))
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lrD,betas=(Dbeta1,Dbeta2))
        self.l2 = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.save_dir = "./models/"
        self.batch_size = batch_size
        self.real = Variable(torch.ones(batch_size,1))
        self.fake = Variable(torch.zeros(batch_size,1))


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))


    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))