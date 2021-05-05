import librosa
import numpy as np
import tensorflow as tf
from glob import glob
from utils.inversion import Inversion_helpers


class Common_helpers():
    def __init__(self, args):
        self.args = args
        self.IH = Inversion_helpers(args)

    # Generate spectrograms from waveform array
    def tospec(self,data):
        specs = np.empty(data.shape[0], dtype=object)
        for i in range(data.shape[0]):
            x = data[i]
            S = self.IH.prep(x)
            S = np.array(S, dtype=np.float32)
            specs[i] = np.expand_dims(S, -1)
        print(specs.shape)
        return specs

    # Generate multiple spectrograms with a determined length from single wav file
    def tospeclong(self, path, length=4*16000):
        x, sr = librosa.load(path, sr=16000)
        x, _ = librosa.effects.trim(x)
        loudls = librosa.effects.split(x, top_db=50)
        xls = np.array([])
        for interv in loudls:
            xls = np.concatenate((xls, x[interv[0]:interv[1]]))
        x = xls
        num = x.shape[0]//length
        specs = np.empty(num, dtype=object)
        for i in range(num-1):
            a = x[i*length:(i+1)*length]
            S = self.IH.prep(a)
            S = np.array(S, dtype=np.float32)
            try:
                sh = S.shape
                specs[i] = S
            except AttributeError:
                print('spectrogram failed')
        print(specs.shape)
        return specs

    # Waveform array from path of folder containing wav files


    def audio_array(self, path):
        ls = glob(f'{path}/*.wav')
        adata = []
        for i in range(len(ls)):
            x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
            x = np.array(x, dtype=np.float32)
            adata.append(x)
        return np.array(adata)

    # Concatenate spectrograms in array along the time axis
    def testass(self, a):
        but = False
        con = np.array([])
        nim = a.self.args.shape[0]
        for i in range(nim):
            im = a[i]
            im = np.squeeze(im)
            if not but:
                con = im
                but = True
            else:
                con = np.concatenate((con, im), axis=1)
        return np.squeeze(con)

    # Split spectrograms in chunks with equal size
    def splitcut(self, data):
        ls = []
        mini = 0
        minifinal = 10*self.args.shape  # max spectrogram length
        for i in range(data.shape[0]-1):
            if data[i].shape[1] <= data[i+1].shape[1]:
                mini = data[i].shape[1]
            else:
                mini = data[i+1].shape[1]
            if mini >= 3*self.args.shape and mini < minifinal:
                minifinal = mini
        for i in range(data.shape[0]):
            x = data[i]
            if x.shape[1] >= 3*self.args.shape:
                for n in range(x.shape[1]//minifinal):
                    ls.append(x[:, n*minifinal:n*minifinal+minifinal, :])
                ls.append(x[:, -minifinal:, :])
        return np.array(ls)
