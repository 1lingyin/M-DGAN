
import torch
import torch.nn as nn
from models.modules.models_small import TF_Domain_Block
from models.modules.models_small import T_Domain_Block


class M_Dnet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(M_Dnet, self).__init__()

        self.n_fft = 400
        self.hop = 100

        self.T_N=T_Domain_Block(in_channels=1,out_channels=1,hidden=60,depth=4,kernel_size=8,stride=4,growth=2,head=1,segment_len=64)
        
        self.TF_N=TF_Domain_Block(num_channel=64, num_features=201)

    def forward(self, x,x_t):

        x_t = self.T_N(x_t)

        est_noisy_spec = torch.stft(x_t.squeeze(1), self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,return_complex=False)
        est_noisy_spec = est_noisy_spec.permute(0, 2, 1, 3)
        est_noisy_real = est_noisy_spec[..., 0].unsqueeze(1)
        est_noisy_imag = est_noisy_spec[..., 1].unsqueeze(1)
        est_noisy_spec = torch.complex(est_noisy_real, est_noisy_imag)
        est_noisy_mag  = torch.abs(est_noisy_spec)
        est_noisy_phase=torch.angle(est_noisy_spec)

        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1)
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        mask,complex_out=self.TF_N(x_in)


        out_mag = mask * mag-est_noisy_mag

        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)

        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)-torch.cos(est_noisy_phase)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)-torch.sin(est_noisy_phase)

        return final_real, final_imag

