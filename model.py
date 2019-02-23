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
    def __init__(self,N_FFT):
        """Docstring for Init of Generator"""
        # We take in a vector of size 1024?
        self.fc1 = nn.Linear(N_FFT,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,N_FFT)


    def forward(self,z):
        """Docstring for forward function"""
        z1 = 1 - z
        x = F.sigmoid(self.fc1(z))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        x1 = F.sigmoid(self.fc1(z1))
        x1 = F.sigmoid(self.fc2(x1))
        x1 = F.sigmoid(self.fc3(x1))
        x1 = F.sigmoid(self.fc4(x1))

        return (x / (x + x1 + np.finfo(float).eps)) * z



    def save(self):
        """Docstring for saving"""


    def load(self):
            """Docstring for loading"""


class Discriminator(nn.Module):
    """Docstring for Discriminator"""

    def __init__(self,in_size):
        """Docstring for Init of Discriminator"""
        self.fc1 = Linear(in_size,513)
        self.fc2 = Linear(513,513)
        self.fc3 = Linear(513,513)
        self.fc4 = Linear(513,513)
        self.fc5 = Linear(513,1)


    def forward(self,z):
        """Docstring for forward function"""
        x = F.sigmoid(self.fc1(z))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x



    def save(self):
        """Docstring for saving"""


    def load(self):
            """Docstring for loading"""


class SVSGan(object):
    """Docstring for SVSGan"""

    def __init__(self,N_FFT,lrG,Gbeta1,Gbeta2,batch_size):
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


    def train(self,step,in_mixture,separated_source):
        """Docstring for training"""

        if step > 1000: # Need intialization Training
            self.gen_optim.zero_grad()
            predicted_source = self.gen(in_mixture)
            G_loss = self.l2(predicted_source,separated_source)
            G_loss.backward()
            self.gen_optim.step()
            
        else: # Regualr GAN Training
            self.dis_optim.zero_grad()
            predicted_source = self.gen(in_mixture)
            
            D_real = self.dis(separated_source)
            D_real_loss = self.bce(D_real,self.real)

            D_fake = self.dis(predicted_source)
            D_fake_loss = self.bce(D_fake,self.fake)

            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            self.dis_optim.step()

            self.gen_optim.zero_grad()
            predicted_source = self.gen(in_mixture)
            D_fake = self.dis(predicted_source)
            G_loss = self.bce(D_fake,self.real)
            G_loss.backward()
            self.gen_optim.step()

        if step%1000 == 0:
            self.save()





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