import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

class Generator(nn.Module):
    """Docstring for Generator"""
    def __init__(self,args):
        super(Generator, self).__init__()
        """Docstring for Init of Generator"""
        self.args = args
        self.fc1 = nn.Conv1d(args.N_FFT,args.ngf,1)
        self.bn1 = nn.BatchNorm1d(args.ngf)
        self.fc2 = nn.Conv1d(args.ngf,args.ngf,1)
        self.bn2 = nn.BatchNorm1d(args.ngf)
        self.fc3 = nn.Conv1d(args.ngf,args.ngf,1)
        self.bn3 = nn.BatchNorm1d(args.ngf)
        self.fc4_1 = nn.Conv1d(args.ngf,args.N_FFT,1)
        self.fc4_2 = nn.Conv1d(args.ngf,args.N_FFT,1)


    def forward(self,z):
        """Docstring for forward function"""
        z = z.float().transpose(1,2).contiguous()
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        vocal = torch.exp(self.fc4_1(x))
        noise = torch.exp(self.fc4_2(x))
        return vocal.transpose(1,2), noise.transpose(1,2)


class Discriminator(nn.Module):
    """Docstring for Discriminator"""

    def __init__(self,args):
        super(Discriminator, self).__init__()
        """Docstring for Init of Discriminator"""
        self.fc1 = nn.Conv1d(args.inD, args.ndf,1)
        self.fc2 = nn.Conv1d(args.ndf, args.ndf,1)
        self.bn2 = nn.BatchNorm1d(args.ndf)
        self.fc3 = nn.Conv1d(args.ndf, args.ndf,1)
        self.bn3 = nn.BatchNorm1d(args.ndf)
        self.fc4 = nn.Conv1d(args.ndf, 1,1)

    def forward(self,x):
        """forward function that takes two sources from the generator"""
        x = x.transpose(1,2).contiguous()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x.transpose(1,2)


class SVSGan(object):
    """Docstring for SVSGan"""

    def __init__(self,args):
        """Docstring for init of SVSGan"""
        self.G = Generator(args).cuda()
        self.D = Discriminator(args).cuda()
        self.gen_optim = optim.Adam(self.G.parameters(),lr=args.lrG,betas=(args.Gbeta1,args.Gbeta2))
        self.dis_optim = optim.Adam(self.D.parameters(),lr=args.lrD,betas=(args.Dbeta1,args.Dbeta2))
        self.l2 = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.save_dir = "./checkpoints/SVSGan"
        self.batch_size = args.batch_size
        self.real = Variable(torch.ones(args.batch_size,args.sample_length , 1)).cuda()
        self.fake = Variable(torch.zeros(args.batch_size,args.sample_length, 1)).cuda()
        self.model_name = "SVSGan"


    def save(self, name):
        """Save the model"""
        save_dir = self.save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        g_state = {
            'gnet' : self.G.state_dict(),
            'gopt' : self.gen_optim.state_dict()
        }
        d_state = {
            'dnet' : self.D.state_dict(),
            'dopt' : self.dis_optim.state_dict()
        }

        torch.save(g_state, os.path.join(save_dir, name + '_G.pkl'))
        torch.save(d_state, os.path.join(save_dir, name + '_D.pkl'))


    def load_G(self, G_checkpoint):
        """Load the generator checkpoints"""
        checkpoint = torch.load(G_checkpoint)
        self.G.load_state_dict(checkpoint['gnet'])
        self.gen_optim.load_state_dict(checkpoint['gopt'])

    def load_D( self, D_checkpoint):
        """Load the discriminator checkpoints"""
        checkpoint = torch.load(D_checkpoint)
        self.D.load_state_dict(checkpoint['dnet'])
        self.dis_optim.load_state_dict(checkpoint['gopt'])



