# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import sys
import time
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from examples.speech_to_speech_translation.models import DurationCodeGenerator
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict

# torch.backends.cudnn.benchmark = True  # Disabled for CPU

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def train(a, h):
    device = torch.device('cpu')  # Force CPU usage
    logger = logging.getLogger()
    
    generator = DurationCodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    logger.info("Checkpoints directory: %s", a.checkpoint_path)
    os.makedirs(a.checkpoint_path, exist_ok=True)

    steps_per_epoch = len(get_dataset_filelist(h)[0]) // h.batch_size
    total_steps = a.training_epochs * steps_per_epoch
    
    state_dict_do = None
    last_epoch = -1
    steps = 0
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        if cp_g and cp_do:
            state_dict_g = load_checkpoint(cp_g, device, map_location=device)
            state_dict_do = load_checkpoint(cp_do, device, map_location=device)
            generator.load_state_dict(state_dict_g['generator'])
            mpd.load_state_dict(state_dict_do['mpd'])
            msd.load_state_dict(state_dict_do['msd'])
            optim_g.load_state_dict(state_dict_g['optim_g'])
            optim_d.load_state_dict(state_dict_do['optim_d'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)
    trainset = CodeDataset(
        training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
        multispkr=h.get('multispkr', None), f0_stats=h.get('f0_stats', None),
        f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
        f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
        vqvae=h.get('code_vq_params', False)
    )
    
    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                             batch_size=h.batch_size, pin_memory=False, drop_last=True)  # Disable pin_memory

    validset = CodeDataset(
        validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
        multispkr=h.get('multispkr', None), f0_stats=h.get('f0_stats', None),
        f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
        f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
        vqvae=h.get('code_vq_params', False)
    )
    validation_loader = DataLoader(validset, num_workers=0, shuffle=False,
                                  batch_size=h.batch_size, pin_memory=False, drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    
    start_epoch = max(0, last_epoch)
    total_epochs = a.training_epochs
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{total_epochs} started")
        
        batch_times = []
        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            x, y, _, y_mel = batch
            y = y.to(device)  # Removed non_blocking
            y_mel = y_mel.to(device)
            y = y.unsqueeze(1)
            x = {k: v.to(device) for k, v in x.items()}

            y_g_hat, dur_losses = generator(**x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, 
                                         h.sampling_rate, h.hop_size, h.win_size,
                                         h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            if h.get('dur_prediction_weight'):
                loss_gen_all += dur_losses * h.dur_prediction_weight
            loss_gen_all.backward()
            optim_g.step()

            steps += 1
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if steps % a.stdout_interval == 0:
                avg_batch_time = sum(batch_times[-10:])/len(batch_times[-10:])
                eta_seconds = avg_batch_time * (len(train_loader) - i)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                
                logger.info(f"Steps: {steps}, Loss: {loss_gen_all.item():.4f}, "
                           f"Batch Time: {batch_time:.2f}s, ETA: {eta_str}")

            if steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps)
                sw.add_scalar("training/mel_spec_error", loss_mel.item(), steps)

        save_checkpoint(
            os.path.join(a.checkpoint_path, f'g_{epoch+1}.pt'),
            {'generator': generator.state_dict(), 'optim_g': optim_g.state_dict()}
        )
        save_checkpoint(
            os.path.join(a.checkpoint_path, f'do_{epoch+1}.pt'),
            {'mpd': mpd.state_dict(), 'msd': msd.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps, 'epoch': epoch}
        )
        logger.info(f"Saved checkpoints for epoch {epoch+1}")

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, batch in enumerate(validation_loader):
                x, y, _, y_mel = batch
                x = {k: v.to(device) for k, v in x.items()}
                y_g_hat, _ = generator(**x)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                             h.sampling_rate, h.hop_size, h.win_size,
                                             h.fmin, h.fmax_for_loss)
                val_loss += F.l1_loss(y_mel.to(device), y_g_hat_mel).item()
        val_loss /= len(validation_loader)
        sw.add_scalar("validation/mel_spec_error", val_loss, epoch)
        logger.info(f"Validation Loss: {val_loss:.4f}")

        scheduler_g.step()
        scheduler_d.step()
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    logger.info("Training complete")

def main():
    logger = setup_logger(os.path.join(os.getcwd(), "logs"))
    logger.info("Initializing Training Process")

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--training_steps', default=62500, type=int)
    parser.add_argument('--stdout_interval', default=10, type=int)
    parser.add_argument('--checkpoint_interval', default=625, type=int)
    parser.add_argument('--summary_interval', default=50, type=int)
    parser.add_argument('--validation_interval', default=625, type=int)
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    config = json.loads(data)
    h = AttrDict(config)
    
    h.num_gpus = 0  # Disable GPU usage
    h.num_workers = 0
    h.dist_config = {}
    
    build_env(a.config, 'config.json', a.checkpoint_path)
    torch.cuda.empty_cache()

    train(a, h)

if __name__ == '__main__':
    main()