import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class STFT(torch.nn.Module):
    def __init__(self, input_data, magnitude, phase, filter_length=1000, hop_length=500):
        super(STFT, self).__init__()
        
        self.input_data = input_data
        self.magnitude = magnitude
        self.phase = phase
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())


    def inverse(self, magnitude, phase):
        num_samples = self.input_data.size(1)

        self.num_samples = num_samples
        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase),
                                               magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        inverse_transform = inverse_transform[:, :, self.filter_length:]
        inverse_transform = inverse_transform[:, :, :self.num_samples]
        return inverse_transform

    def forward(self):
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction