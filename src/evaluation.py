import numpy as np
from models import generator
from natsort import natsorted
import os
from models.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm

@torch.no_grad()
def enhance_one_track(model, audio_path,cut_len, n_fft=400, hop=100):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len/cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,return_complex=False)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec,noisy.unsqueeze(1))

    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = torch.view_as_complex(power_uncompress(est_real, est_imag).squeeze(1))
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).cuda(),
                            onesided=True,return_complex=False)

    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length

    if not os.path.exists("./enhanced voice"):
        os.mkdir("./enhanced voice")
    saved_path = os.path.join("./enhanced voice", name)
    sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir):
    n_fft = 400
    model = generator.M_Dnet(num_channel=64, num_features=n_fft//2+1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()
    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    loop = tqdm(enumerate(audio_list), total=len(audio_list), ncols=100)
    for idx,audio in loop:
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, 16000*16, n_fft, n_fft//4)
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='Model Path')
parser.add_argument("--test_dir", type=str, default='VCTK-DEMAND test dataset dir')

args = parser.parse_args()


if __name__ == '__main__':
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    evaluation(args.model_path, noisy_dir, clean_dir)
