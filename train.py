import numpy as np
import tqdm
import time
import os
import utils
from model import SVSGan
from tensorboardX import SummaryWriter
import torch
from torch.autograd import grad as torch_grad

# from graphviz import Digraph
import torch
from torch.autograd import Variable


# make_dot was moved to https://github.com/szagoruyko/pytorchviz
# from torchviz import make_dot

def _gradient_penalty_centered_(c_real, model, gp_weight, center=0.):
    B = c_real[0].size(0)
    #print(c_real[0].requires_grad)
    c_real[0].requires_grad_(True)
    #c_real[1].requires_grad_(True)
    # Calculate gradients of probabilities with respect to examples
    #make_dot(d).view()
    d = model.D(c_real[0], c_real[1])
    gradients = torch_grad(
            outputs=d, inputs=c_real[0],
            grad_outputs=torch.ones(d.size()).cuda(),
            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (B, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.contiguous().view(B, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - center) ** 2).mean()


def gan_training(model, step, vocal_real, bgm_real, in_mixture, phase, size, gp_center, writer):
    model.dis_optim.zero_grad()

    vocal_fake, bgm_fake = model.G(in_mixture)

    D_real = model.D(vocal_real)
    D_real_loss = model.bce(D_real, model.real[:D_real.shape[0], :, :])
    gp = _gradient_penalty_centered_([vocal_real, bgm_real], model, args.gp_weight, center=gp_center)

    D_fake = model.D(vocal_fake)
    D_fake_loss = model.bce(D_fake, model.fake[:D_fake.shape[0],:,:])

    D_loss = D_real_loss + D_fake_loss + gp
    writer.add_scalar('D_loss', D_loss.cpu().detach().item(), step)
    D_loss.backward()
    model.dis_optim.step()

    model.gen_optim.zero_grad()
    vocal_fake, bgm_fake = model.G(in_mixture)

    D_fake = model.D(vocal_fake)

    vocal_fake_wav = utils.reConstructWav(size, vocal_fake.cpu().detach().numpy(), phase)
    bgm_fake_wav = utils.reConstructWav(size, bgm_fake.cpu().detach().numpy(), phase)
    mixture_fake_wav = vocal_fake_wav + bgm_fake_wav
    mixture_wav = utils.reConstructWav(size, in_mixture.cpu().detach().numpy(), phase)

    rec_loss = model.l2(mixture_wav, mixture_fake_wav)

    G_loss = model.bce(D_fake, model.real[:D_fake.shape[0], :, :])
    G_loss = G_loss + rec_loss
    writer.add_scalar('G_loss', G_loss.cpu().detach().item(), step)
    G_loss.backward()
    model.gen_optim.step()
    return D_loss, G_loss, gp

def train_gan(model, data_loader, epochs, args, writer):

    step = 0
    pbar = tqdm.trange(epochs, leave=False)
    for epoch in pbar:
        # for batch_i, (in_mixture_dic, vocal_real_dic, bgm_real_dic) in tqdm.tqdm(enumerate(data_loader), leave=False):
        for batch_i, (in_mixture_dic, vocal_real_dic, bgm_real_dic) in enumerate(data_loader):
            vocal_real = vocal_real_dic['magnitude'].float().cuda()
            bgm_real = bgm_real_dic['magnitude'].float().cuda()
            in_mixture = in_mixture_dic['magnitude'].float().cuda()
            mixture_phase = in_mixture_dic['phase'].float().cuda()
            size = in_mixture_dic['size']

            D_loss, G_loss, gp = gan_training(model, step, vocal_real, bgm_real, in_mixture, mixture_phase, size, args.gp_center, writer)
            step += 1

        if (epoch) % 10  == 0:
            # print("Epoch [% 2d/% 2d] D_Loss %2.5f G_Loss %2.5f"\
            #           %(epoch, epochs, D_loss, G_loss))
            pbar.set_description("Epoch [% 2d/% 2d] D_Loss %2.5f G_Loss %2.5f gp %2.5f"\
                      %(epoch, epochs, D_loss, G_loss, gp))


            #if step % 1000 == 0:
            #    model.save('checkpoint_'+ str(step))
                # TODO: eval


if __name__ == '__main__':
    args = utils.get_args()

    batch_size = args.batch_size;
    # Continue placing other arguments here

    model = SVSGan(args)

    if args.resume_G != None:
        model.load_G(args.resume_G)

    if args.resume_D != None:
        model.load_D(args.resume_D)

    data_loader = utils.get_loader(args)

    # Writer, and save directory
    ts = int(time.time())
    if args.log_dir is None:
        args.log_dir = "run-%d"%ts
    else:
        args.log_dir = "%s-%d"%(args.log_dir, ts)
    writer = SummaryWriter(log_dir=os.path.join("runs", "train", args.log_dir))
    train_gan(model, data_loader['train'], 10000, args, writer)



