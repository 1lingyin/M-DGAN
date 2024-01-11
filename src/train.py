import argparse
# log
import logging
import os
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.modules import dataloader
from models import discriminator
from models.generator import M_Dnet
from utils import *

# 可视化操作
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer = SummaryWriter(log_dir='/root/tf-logs/V3_1/' + 'Time_' + TIMESTAMP, flush_secs=180)
logging.basicConfig(level=logging.INFO)

# 3.0版本
# 使用了优化conformer,混合方式为mag中混合,采用自适应的学习率
# 参数[0.4, 0.6, 0.9, 0.07]
# 3.1版本是在3.0的基础上更改了自适应学习率参数,更改了注意力
parser = argparse.ArgumentParser()
parser.add_argument("--loss_weights", type=list, default=[0.4, 0.6, 0.9, 0.07])
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--data_dir", type=str, default='VCTK-DEMAND dataset dir')
parser.add_argument("--save_model_dir", type=str, default='./epoch_models')

args = parser.parse_args()


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 400
        self.hop = 100
        self.train_dataset = train_ds
        self.test_dataset = test_ds
        self.model = M_Dnet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()

        #计算参量数
        params = list(self.model.parameters())
        k = 0
        for i in params:
            l = 1
            print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：" + str(k))

        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2 * 5e-4)

    def compute_loss(self,est_real, est_imag,clean_real, clean_imag,est_audio,clean):
        one_labels = torch.ones(args.batch_size).cuda()

        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        time_loss = torch.mean(torch.abs(est_audio - clean))

        return args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
            + args.loss_weights[3] * gen_loss_GAN
    def train_one_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()

        self.optimizer.zero_grad()

        #预处理
        noisy, clean, noisy_spec, clean_spec, clean_real, clean_imag=preprocess(noisy,clean,self.n_fft,self.hop)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        est_real, est_imag = self.model(noisy_spec, noisy.unsqueeze(1))

        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        est_spec_uncompress = torch.view_as_complex(power_uncompress(est_real, est_imag).squeeze(1))
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)
        length = est_audio.size(-1)

        #计算LOSS
        loss=self.compute_loss(est_real,est_imag,clean_real,clean_imag,est_audio,clean)
        loss.backward()
        self.optimizer.step()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])

        pesq_score, pesq_avg = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        one_labels = torch.ones(args.batch_size).cuda()
        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        pesq_total = 0.
        for idx, batch in enumerate(self.test_dataset):
            step = idx + 1
            loss, disc_loss, pesq_avg = self.test_one_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
            pesq_total += pesq_avg
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / step

        Epochtestinfo = 'Generator average loss: {}, Discriminator average loss: {}'
        logging.info(
            Epochtestinfo.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg, disc_loss_avg, pesq_avg

    @torch.no_grad()
    def test_one_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(args.batch_size).cuda()

        #预处理
        noisy, clean, noisy_spec, clean_spec, clean_real, clean_imag=preprocess(noisy,clean,self.n_fft,self.hop)

        est_real, est_imag = self.model(noisy_spec, noisy.unsqueeze(1))

        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
        est_spec_uncompress = torch.view_as_complex(power_uncompress(est_real, est_imag).squeeze(1))
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)
        length = est_audio.size(-1)

        #计算LOSS
        loss=self.compute_loss(est_real,est_imag,clean_real,clean_imag,est_audio,clean)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])

        pesq_score, pesq_avg = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            pesq_avg = 0
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, verbose=True,
                                                                 patience=2, threshold_mode='rel', cooldown=0, min_lr=0,
                                                                 eps=1e-08)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_disc, mode="min", factor=0.45,
                                                                 verbose=True, patience=2, threshold_mode='rel',
                                                                 cooldown=0, min_lr=0, eps=1e-08)
        total_step = 0
        for epoch in range(args.epochs):
            self.model.train()
            self.discriminator.train()
            loop = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset), ncols=100)
            for batch_num, batch in loop:
                # 训练一个step
                loss, disc_loss = self.train_one_step(batch)

                # 更新信息
                # tqdm&tensorborad
                loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
                loop.set_postfix(g_loss=loss, scores_loss=disc_loss)
                total_step += 1
                writer.add_scalar('G_Loss_BY_step', loss, total_step)
                writer.add_scalar('D Loss_BY_step', disc_loss, total_step)
                writer.flush()

            gen_loss, dic_loss, pesq_avg = self.test()
            path = os.path.join(args.save_model_dir, 'epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            torch.save(self.model.state_dict(), path)
            scheduler_G.step(gen_loss)
            scheduler_D.step(dic_loss)

            # 更新信息log
            writer.add_scalar('G_LR', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            writer.add_scalar('D_LR', self.optimizer_disc.state_dict()['param_groups'][0]['lr'], epoch)
            writer.add_scalar('PESQ_AVG', pesq_avg, epoch)
            writer.add_scalar('GLoss_BYepoch', loss, epoch)
            writer.add_scalar('DLoss_BYepoch', disc_loss, epoch)
            writer.add_scalar('TEST_Gloss_BYepoch', gen_loss, epoch)
            writer.add_scalar('TEST_Dloss_BYepoch', dic_loss, epoch)
            writer.flush()


def main():

    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 2, 16000 * 2)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    main()
