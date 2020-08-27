import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str
from utils.pqmf import PQMF
from denoiser import Denoiser

MAX_WAV_VALUE = 32768.0


def pad_tensor(x, pad, side='both'):
  # NB - this is just a quick method i need right now
  # i.e., it won't generalise to other shapes/dims
  b, t, c = x.shape
  total = t + 2 * pad if side == 'both' else t + pad
  padded = torch.zeros(b, total, c).cuda()
  if side == 'before' or side == 'both' :
      padded[:, pad:pad+t, :] = x
  elif side == 'after':
      padded[:, :t, :] = x
  return padded


def fold_with_overlap(x, target, overlap) :

  ''' Fold the tensor with overlap for quick batched inference.
      Overlap will be used for crossfading in xfade_and_unfold()
  Args:
      x (tensor)    : Upsampled conditioning features.
                      shape=(1, timesteps, features)
      target (int)  : Target timesteps for each index of batch
      overlap (int) : Timesteps for both xfade and rnn warmup
  Return:
      (tensor) : shape=(num_folds, target + 2 * overlap, features)
  Details:
      x = [[h1, h2, ... hn]]
      Where each h is a vector of conditioning features
      Eg: target=2, overlap=1 with x.size(1)=10
      folded = [[h1, h2, h3, h4],
                [h4, h5, h6, h7],
                [h7, h8, h9, h10]]
  '''

  _, total_len, features = x.shape

  # Calculate variables needed
  num_folds = (total_len - overlap) // (target + overlap)
  extended_len = num_folds * (overlap + target) + overlap
  remaining = total_len - extended_len

  # Pad if some time steps poking out
  if remaining != 0 :
      num_folds += 1
      padding = target + 2 * overlap - remaining
      x = pad_tensor(x, padding, side='after')

  folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()

  # Get the values for the folded tensor
  for i in range(num_folds) :
      start = i * (target + overlap)
      end = start + target + 2 * overlap
      folded[i] = x[:, start:end, :]

  return folded


def xfade_and_unfold(y, target, overlap) :

  ''' Applies a crossfade and unfolds into a 1d array.
  Args:
      y (ndarry)    : Batched sequences of audio samples
                      shape=(num_folds, target + 2 * overlap)
                      dtype=np.float64
      overlap (int) : Timesteps for both xfade and rnn warmup
  Return:
      (ndarry) : audio samples in a 1d array
                  shape=(total_len)
                  dtype=np.float64
  Details:
      y = [[seq1],
            [seq2],
            [seq3]]
      Apply a gain envelope at both ends of the sequences
      y = [[seq1_in, seq1_target, seq1_out],
            [seq2_in, seq2_target, seq2_out],
            [seq3_in, seq3_target, seq3_out]]
      Stagger and add up the groups of samples:
      [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
  '''

  num_folds, length = y.shape
  target = length - 2 * overlap
  total_len = num_folds * (target + overlap) + overlap

  # Need some silence for the rnn warmup
  silence_len = overlap // 2
  fade_len = overlap - silence_len
  silence = np.zeros((silence_len))

  # Equal power crossfade
  t = np.linspace(-1, 1, fade_len)
  fade_in = np.sqrt(0.5 * (1 + t))
  fade_out = np.sqrt(0.5 * (1 - t))

  # Concat the silence to the fades
  fade_in = np.concatenate([silence, fade_in])
  fade_out = np.concatenate([fade_out, silence])

  # Apply the gain to the overlap samples
  y[:, :overlap] *= fade_in
  y[:, -overlap:] *= fade_out

  unfolded = np.zeros((total_len))

  # Loop to add up all the samples
  for i in range(num_folds ) :
      start = i * (target + overlap)
      end = start + target + 2 * overlap
      unfolded[start:end] += y[i]

  return unfolded


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)

    with torch.no_grad():
        mel = torch.from_numpy(np.load(args.input))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        mel = mel.view(1, -1, 80)
        mel = fold_with_overlap(mel, target = 2, overlap = 1)
        print(mel.shape, "Shape of mel after fold with overlap")   #n_fold, 4, 80
        num_folds, column = mel.shape[0] , mel.shape[1]
        y = []
        for i in range(0, num_folds):
            input_mel = mel[i,:,:]
            input_mel = input_mel.squeeze(0)
            audio = model.inference(input_mel)
            y.append(audio)
        y = np.array(y)
        print(y.shape,"Shape of y before passing into xfade_and_unfold")  #n_fold, 4
        y = y.reshape(num_folds, column)
        audio = xfade_and_unfold(y, target = 2, overlap = 1)
        audio = audio.unsqueeze(0)
        # For multi-band inference
        if hp.model.out_channels > 1:
            pqmf = PQMF()
            audio = pqmf.synthesis(audio).view(-1)
        audio = audio.squeeze(0)  # collapse all dimension except time axis
        if args.d:
            denoiser = Denoiser(model).cuda()
            audio = denoiser(audio, 0.1)
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        audio = audio.cpu().detach().numpy()

        out_path = args.input.replace('.npy', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
        write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    parser.add_argument('-d', action='store_true', help="denoising ")
    args = parser.parse_args()

    main(args)
