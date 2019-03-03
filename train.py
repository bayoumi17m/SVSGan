import numpy as np
from tensorboardX import SummaryWriter
import torch


def _gradient_penalty_centered_(self, c_real, gp_weight, center=0.):
    B = c_real.size(0)
    d = self.d_net(c_real)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(
            outputs=d, inputs=c_real,
            grad_outputs=torch.ones(d.size()).cuda(),
            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (B, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(B, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - center) ** 2).mean()


def tf_masking(source, in_mixture):
    return source * in_mixture


def gan_training(model, step, vocal_real,bgm_real,in_mixture, gp_center, writer):
    model.dis_optim.zero_grad()
    y1 = model.G(in_mixture)
    y2 = torch.ones_like(y1) - y1
    vocal_fake = tf_masking(y1, in_mixture)
    bgm_fake = tf_masking(y2, in_mixture)

    D_real = model.D(vocal_real, bgm_real, in_mixture)
    D_real_loss = model.bce(D_real, model.real)

    D_fake = model.D(vocal_fake, bgm_fake, in_mixture)
    D_fake_loss = model.bce(D_fake, model.fake)

    gp = self._gradient_penalty_centered_(c_real, center=gp_center)
    D_loss = D_real_loss + D_fake_loss + gp
    writer.add_scaler('D_loss', D_loss, step)
    D_loss.backward()
    model.dis_optim.step()

    model.gen_optim.zero_grad()
    y1 = model.G(in_mixture)
    y2 = torch.ones_like(y1) - y1
    vocal_fake = tf_masking(y1, in_mixture)
    bgm_fake = tf_masking(y2, in_mixture)

    D_fake = model.D(vocal_fake, bgm_fake, in_mixture)
    G_loss = model.bce(D_fake, model.real)
    writer.add_scaler('G_loss', G_loss, step)
    G_loss.backward()
    model.gen_optim.step()


def train_gan(model, data_loader, epochs):

    step = 0
    for epoch in range(epochs):
        for in_mixture_dic, vocal_real_dic, bgm_real_dic in data_loader:
            vocal_real = vocal_real_dic['magnitude'].cuda()
            bgm_real = bgm_real_dic['magnitude'].cuda()
            in_mixture = sin_mixture_dic['magnitude'].cuda()

            gan_training(model, step, vocal_real,bgm_real,in_mixture)

            if step % 1000 == 0:
                model.save()
                # TODO: eval

            step += 1


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


    writer = SummaryWriter()
    train_gan(model, data_loader, 10000, writer)



