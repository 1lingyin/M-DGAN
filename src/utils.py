import torch
import torch.nn as nn


#每条数据预处理
def preprocess(noisy,clean,n_fft,hop):
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)
    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(),
                                    onesided=True, return_complex=False)
    clean_spec = torch.stft(clean, n_fft, hop, window=torch.hamming_window(n_fft).cuda(),
                                    onesided=True, return_complex=False)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

    return noisy, clean,noisy_spec,clean_spec,clean_real,clean_imag



def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


