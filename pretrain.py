import numpy as np
import tqdm
import time
import torch
import utils
import model
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint
from tensorboardX import SummaryWriter
import os
from scipy import signal

def step(model, opt, data, step, writer, args):
    #print(data)
    mixture, vocal, noise = data

    mix_spec = mixture['magnitude'].float() + np.finfo(np.float64).eps
    voc_spec = vocal['magnitude'].float()
    noi_spec = noise['magnitude'].float()

    delta = args.delta
    #noi_spec[np.where(np.abs(noi_spec) < delta)] = 0.0
    #voc_spec[np.where(np.abs(voc_spec) < delta)] = 0.0

    if args.cuda:
        mix_spec, voc_spec, noi_spec = mix_spec.cuda(), voc_spec.cuda(), noi_spec.cuda()
    #print("max voc_spec: " + str(voc_spec.max()))
    #print("min mix_spec: " + str(voc_spec.min()))
    #print('max:' + str((voc_spec / mix_spec).max()))
    #print('min:' + str((voc_spec / mix_spec).min()))
    vocal_recon, noise_recon = model(mix_spec)

    noise_recon = torch.where(torch.abs(mix_spec) >= delta, noise_recon, torch.zeros_like(noise_recon))
    vocal_recon = torch.where(torch.abs(mix_spec) >= delta, noise_recon, torch.zeros_like(vocal_recon))

    vocal_recon_loss = F.mse_loss(vocal_recon/mix_spec, voc_spec/mix_spec)
    noise_recon_loss = F.mse_loss(noise_recon/mix_spec, noi_spec/mix_spec)
    loss = args.vocal_recon_weight * vocal_recon_loss + args.noise_recon_weight * noise_recon_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    # LOG
    writer.add_scalar('pretrain/vocal_loss', vocal_recon_loss.cpu().detach().item(), step)
    writer.add_scalar('pretrain/noise_loss', noise_recon_loss.cpu().detach().item(), step)
    writer.add_scalar('pretrain/loss', loss.cpu().detach().item(), step)

    return vocal_recon_loss.cpu().detach().item(), noise_recon_loss.cpu().detach().item(), loss.cpu().detach().item()


def validate(model, loader, epoch, writer, args):
    with torch.no_grad():
        vocal_recon_loss = 0.
        noise_recon_loss = 0.
        n = 0
        idx = np.random.randint(len(loader))
        for i, data in enumerate(loader):
            mixture, vocal, noise = data
            mix_spec = mixture['magnitude'].float()
            voc_spec = vocal['magnitude'].float()
            noi_spec = noise['magnitude'].float()
            if args.cuda:
                mix_spec, voc_spec, noi_spec = mix_spec.cuda(), voc_spec.cuda(), noi_spec.cuda()
            vocal_recon, noise_recon = model(mix_spec)

            noise_recon_loss += F.mse_loss(noise_recon/mix_spec, noi_spec/mix_spec)
            vocal_recon_loss += F.mse_loss(vocal_recon/mix_spec, voc_spec/mix_spec)
            n += mix_spec.size(0)

            if epoch % 50 == 0 and i == idx:
                idx_b = np.random.randint(mix_spec.shape[0])
#                 writer.add_image("val/mix_spec", mix_spec[idx_b].cpu().detach(), epoch)
                writer.add_image("val/voc_spec", voc_spec[idx_b].cpu().detach(), epoch)
                writer.add_image("val/noi_spec", noi_spec[idx_b].cpu().detach(), epoch)

                writer.add_image("val/vocal_recon", vocal_recon[idx_b].cpu().detach(), epoch)
                writer.add_image("val/noise_recon", noise_recon[idx_b].cpu().detach(), epoch)

                _ , mixtlabel = signal.istft(mix_spec[idx_b].cpu().detach().numpy().T * np.exp(1j * mixture['phase'][idx_b].cpu().detach().numpy().T), fs = 16000)
                _ , label = signal.istft(voc_spec[idx_b].cpu().detach().numpy().T * np.exp(1j * vocal['phase'][idx_b].cpu().detach().numpy().T), fs = 16000)
                _ , recon = signal.istft(np.array(vocal_recon[idx_b].cpu().detach().numpy().T * np.exp(1j * vocal['phase'][idx_b].cpu().detach().numpy().T)), fs = 16000)

                writer.add_audio("val/vocal_mixture_audio", mixtlabel/ np.abs(mixtlabel).max() ,epoch, sample_rate =  16000)
                writer.add_audio("val/vocal_label_audio", label/ np.abs(label).max() ,epoch, sample_rate =  16000)
                writer.add_audio("val/vocal_recon_audio", recon/ np.abs(recon).max() ,epoch, sample_rate =  16000)

        n = float(n)
        vocal_recon_loss /= n
        noise_recon_loss /= n

        # Log
        writer.add_scalar("val/vocal_loss", vocal_recon_loss.cpu().detach().item(), epoch)
        writer.add_scalar("val/noise_loss", noise_recon_loss.cpu().detach().item(), epoch)

        utils.log_score(args.metric_directory, model, args.sample_length, epoch, writer)

    return {
        'vocal' : vocal_recon_loss.cpu().detach().item(),
        'noise' : vocal_recon_loss.cpu().detach().item()
    }


def save(model, opt, save_dir, epoch, args):
    os.makedirs(save_dir, exist_ok=True)

    # Save the current epoch
    save_filename = os.path.join(save_dir, "checkpoint-%d.pt"%epoch)
    torch.save({
        'model' : model.state_dict(),
        'opt'   : opt.state_dict()
    }, save_filename)

    # Save the most recent
    save_filename = os.path.join(save_dir, "checkpoint.pt")
    torch.save({
        'model' : model.state_dict(),
        'opt'   : opt.state_dict()
    }, save_filename)


def resume(ckpt_path, model, opt):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    opt = optim.Adam(model.parameters(), lr=args.lrG, betas=(args.Gbeta1,args.Gbeta2))
    opt.load_state_dict(ckpt['opt'])
    return model, opt


if __name__ == "__main__":
    args = utils.get_args()

    # Models and optimizers
    model = model.Generator(args)
    if args.cuda:
        model.cuda()
    opt = optim.Adam(model.parameters(), lr=args.lrG, betas=(args.Gbeta1,args.Gbeta2))

    # Resume
    if args.resume is not None:
        model, opt = resume(args.resume, model, opt)


    # Data loaders
    loaders = utils.get_loader(args)
    tr_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']

    # Writer, and save directory
    ts = int(time.time())
    if args.log_dir is None:
        args.log_dir = "run-%d"%ts
    else:
        args.log_dir = "%s-%d"%(args.log_dir, ts)

    writer = SummaryWriter(log_dir=os.path.join("runs", "pretrain", args.log_dir))
    save_dir =  os.path.join("checkpoints", "pretrain", args.log_dir)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    for epoch_idx in range(args.epochs):
        epoch = start_epoch + epoch_idx
        epoch_loss = 0
        epoch_vocal_recon_loss = 0
        epoch_noise_recon_loss = 0
        for b_idx, data in tqdm.tqdm(enumerate(tr_loader)):
            step_num = epoch_idx * len(tr_loader) + b_idx
            vocal_l, noise_l, loss = step(model, opt, data, step_num, writer, args)
            epoch_vocal_recon_loss += vocal_l
            epoch_noise_recon_loss += noise_l
            epoch_loss +=  loss

            if (b_idx + 1) % args.log_step == 0:
                print("Epoch [% 2d/% 2d] Batch [% 2d/% 2d] Loss %2.5f"\
                      %(epoch, args.epochs, b_idx, len(tr_loader), loss))

        print("Epoch loss for epoch idx:%s"%epoch_idx)
        print(epoch_loss)

        writer.add_scalar('pretrain/epoch_vocal_loss', epoch_vocal_recon_loss, epoch_idx)
        writer.add_scalar('pretrain/epoch_noise_loss', epoch_noise_recon_loss, epoch_idx)
        writer.add_scalar('pretrain/epoch_loss', epoch_loss, epoch_idx)


        if (epoch_idx + 1) % args.val_freq == 0:
            print("Val for epoch idx: %s"%epoch_idx)
            res = validate(model, val_loader, epoch, writer, args)
            print(res)


        if (epoch_idx + 1) % args.save_freq == 0:
            save(model, opt, save_dir, epoch, args)


