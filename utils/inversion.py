import torch
import torch.nn as nn
from tqdm import tqdm
from torchaudio.transforms import MelScale, Spectrogram
import librosa
import numpy as np

class Inversion_helpers():
    def __init__(self, args):
        self.args = args
        if args.device == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        specobj = Spectrogram(n_fft=6 * self.args.hop, win_length=6 * self.args.hop, hop_length=self.args.hop, pad=0, power=2, normalized=True)
        self.specfunc = specobj.forward
        melobj = MelScale(n_mels=self.args.hop, sample_rate=self.args.sr, f_min=0.)
        self.melfunc = melobj.forward

    def melspecfunc(self, waveform):
      specgram = self.specfunc(waveform)
      mel_specgram = self.melfunc(specgram)
      return mel_specgram

    def spectral_convergence(self, input, target):
        return 20 * ((input - target).norm().log10() - target.norm().log10())

    def GRAD(self, spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.003):

        spec = torch.Tensor(spec)
        samples = (spec.shape[-1]*self.args.hop)-self.args.hop

        if init_x0 is None:
            init_x0 = spec.new_empty((1, samples)).normal_(std=1e-6)
        x = nn.Parameter(init_x0)
        T = spec

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam([x], lr=lr)

        bar_dict = {}
        metric_func = self.spectral_convergence
        bar_dict['spectral_convergence'] = 0
        metric = 'spectral_convergence'

        init_loss = None
        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for i in range(maxiter):
                optimizer.zero_grad()
                V = transform_fn(x)
                loss = criterion(V, T)
                loss.backward()
                optimizer.step()
                lr = lr*0.9999
                for param_group in optimizer.param_groups:
                  param_group['lr'] = lr

                if i % evaiter == evaiter - 1:
                    with torch.no_grad():
                        V = transform_fn(x)
                        bar_dict[metric] = metric_func(V, spec).item()
                        l2_loss = criterion(V, spec).item()
                        pbar.set_postfix(**bar_dict, loss=l2_loss)
                        pbar.update(evaiter)

        return x.detach().view(-1).cpu()

    def normalize(self, S):
      return np.clip((((S - self.args.min_level_db) / -self.args.min_level_db)*2.)-1., -1, 1)

    def denormalize(self, S):
      return (((np.clip(S, -1, 1)+1.)/2.) * -self.args.min_level_db) + self.args.min_level_db

    def prep(self, wv):
      S = np.array(torch.squeeze(self.melspecfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
      S = librosa.power_to_db(S)-self.args.ref_level_db
      return self.normalize(S)

    def deprep(self,S):
      S = self.denormalize(S)+self.args.ref_level_db
      S = librosa.db_to_power(S)
      wv = self.GRAD(np.expand_dims(S,0), self.melspecfunc, maxiter=2000, evaiter=10, tol=1e-8)
      return np.array(np.squeeze(wv))