import numpy as np
import torch

def tf_masking(source, in_mixture):
    return source * in_mixture


def initialize_training(model,vocal_real,bgm_real,in_mixture):
    model.gen_optim.zero_grad()
    y1 = model.G(in_mixture)
    y2 = torch.ones_like(y1) - y1
    vocal_fake = tf_masking(y1, in_mixture)
    bgm_fake = tf_masking(y2, in_mixture)

    G_loss = model.l2(vocal_real, vocal_fake) + model.l2(bgm_real, bgm_fake)
    G_loss.backward()
    model.gen_optim.step()


def gan_training(model,vocal_real,bgm_real,in_mixture):
    model.dis_optim.zero_grad()
    y1 = model.G(in_mixture)
    y2 = torch.ones_like(y1) - y1
    vocal_fake = tf_masking(y1, in_mixture)
    bgm_fake = tf_masking(y2, in_mixture)
    
    D_real = model.D(vocal_real, bgm_real, in_mixture)
    D_real_loss = model.bce(D_real, model.real)

    D_fake = model.D(vocal_fake, bgm_fake, in_mixture)
    D_fake_loss = model.bce(D_fake, model.fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    model.dis_optim.step()

    model.gen_optim.zero_grad()
    y1 = model.G(in_mixture)
    y2 = torch.ones_like(y1) - y1
    vocal_fake = tf_masking(y1, in_mixture)
    bgm_fake = tf_masking(y2, in_mixture)

    D_fake = model.D(vocal_fake, bgm_fake, in_mixture)
    G_loss = model.bce(D_fake, model.real)
    G_loss.backward()
    model.gen_optim.step()


def train_overall(model,step,vocal_real,bgm_real,in_mixture):
        """Docstring for training"""

        if step <= 1000: # Need intialization Training
            initialize_training(model,vocal_real,bgm_real,in_mixture)
            
        else: # Regualr GAN Training
            gan_training(model,vocal_real,bgm_real,in_mixture)

        if step % 1000 == 0:
            model.save()