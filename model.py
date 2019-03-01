import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os
import utils

class Generator(nn.Module):
    """Docstring for Generator"""
    def __init__(self,args):
        super(Generator, self).__init__()
        """Docstring for Init of Generator"""
        # We take in a vector of size 1024?
        self.fc1 = nn.Linear(args.N_FFT,args.ngf)
        self.bn1 = nn.BatchNorm1d(args.ngf)
        self.fc2 = nn.Linear(args.ngf,args.ngf)
        self.bn2 = nn.BatchNorm1d(args.ngf)
        self.fc3 = nn.Linear(args.ngf,args.ngf)
        self.bn3 = nn.BatchNorm1d(args.ngf)
        self.fc4 = nn.Linear(args.ngf,args.N_FFT)


    def forward(self,z):
        """Docstring for forward function"""
        #z1 = 1 - z
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.tanh(self.fc4(x))

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

    def __init__(self,args):
        super(Discriminator, self).__init__()
        """Docstring for Init of Discriminator"""
        self.fc1 = nn.Linear(args.inD, args.ndf)
        self.fc2 = nn.Linear(args.ndf, args.ndf)
        self.bn2 = nn.BatchNorm1d(args.ndf)
        self.fc3 = nn.Linear(args.ndf, args.ndf)
        self.bn3 = nn.BatchNorm1d(args.ndf)
        self.fc4 = nn.Linear(args.ndf, 1)

    def forward(self,y1,y2,z):
        """forward function that takes two sources from the generator and the mixture"""
        x = torch.cat([y1, y2, z], 1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))

        return x
    #     self.fc1 = Linear(in_size,hidden_nodes_D)
    #     self.fc2 = Linear(hidden_nodes_D,hidden_nodes_D)
    #     self.fc3 = Linear(hidden_nodes_D,hidden_nodes_D)
    #     self.fc4 = Linear(hidden_nodes_D,1)
        
    #     self.bn1 = nn.BatchNorm(batch_size)
    #     self.bn2 = nn.BatchNorm(batch_size)
    #     self.bn3 = nn.BatchNorm(batch_size)
    #     self.bn4 = nn.BatchNorm(batch_size)



    # def forward(self,z):
    #     """Docstring for forward function"""
    #     x = F.relu(self.bn1(self.fc1(z)))
    #     x = F.relu(self.bn2(self.fc2(x)))
    #     x = F.relu(self.bn3(self.fc3(x)))
    #     x = F.sigmoid(self.bn4(self.fc4(x)))
    #     return x



    def save(self):
        """Docstring for saving"""


    def load(self):
            """Docstring for loading"""


class SVSGan(object):
    """Docstring for SVSGan"""

    def __init__(self,args):
        """Docstring for init of SVSGan"""
        self.G = Generator(args)
        self.D = Discriminator(args)
        self.gen_optim = optim.Adam(self.G.parameters(),lr=args.lrG,betas=(args.Gbeta1,args.Gbeta2))
        self.dis_optim = optim.Adam(self.D.parameters(),lr=args.lrD,betas=(args.Dbeta1,args.Dbeta2))
        self.l2 = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.save_dir = "./models/"
        self.batch_size = args.batch_size
        self.real = Variable(torch.ones(args.batch_size,1))
        self.fake = Variable(torch.zeros(args.batch_size,1))
        self.model_name = "SVSGan"


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


