import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import os



def initialize_training(model,in_mixture,separated_source):
    model.gen_optim.zero_grad()
    predicted_source = model.gen(in_mixture)
    G_loss = model.l2(predicted_source,separated_source)
    G_loss.backward()
    model.gen_optim.step()


def gan_training(model,step,in_mixture,separated_source):
    model.dis_optim.zero_grad()
    predicted_source = model.gen(in_mixture)
    
    D_real = model.dis(separated_source)
    D_real_loss = model.bce(D_real,model.real)

    D_fake = model.dis(predicted_source)
    D_fake_loss = model.bce(D_fake,model.fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    model.dis_optim.step()

    model.gen_optim.zero_grad()
    predicted_source = model.gen(in_mixture)
    D_fake = model.dis(predicted_source)
    G_loss = model.bce(D_fake,model.real)
    G_loss.backward()
    model.gen_optim.step()


def train_overall(model,step,in_mixture,separated_source):
        """Docstring for training"""

        if step > 1000: # Need intialization Training
            initialize_training(model,in_mixture,separated_source)
            
        else: # Regualr GAN Training
            gan_training(model,in_mixture,separated_source)

        if step%1000 == 0:
            model.save()