import numpy as np
import soundfile as sf
import os
from utils.inversion import Inversion_helpers
import matplotlib.pyplot as plt
import IPython


# After Training, use these functions to convert data with the generator and save the results
class After_Training_helpers():
    def __init__(self, args, gen):
        self.args = args
        self.gen = gen

    # Assembling generated Spectrogram chunks into final Spectrogram
    def specass(self, a, spec, shape):
        but = False
        con = np.array([])
        nim = a.shape[0]
        for i in range(nim-1):
            im = a[i]
            im = np.squeeze(im)
            if not but:
                con = im
                but = True
            else:
                con = np.concatenate((con, im), axis=1)
        diff = spec.shape[1]-(nim*shape)
        a = np.squeeze(a)
        con = np.concatenate((con, a[-1, :, -diff:]), axis=1)
        return np.squeeze(con)

    # Splitting input spectrogram into different chunks to feed to the generator
    def chopspec(self, spec, shape):
        dsa = []
        for i in range(spec.shape[1]//shape):
            im = spec[:, i*shape:i*shape+shape]
            im = np.reshape(im, (im.shape[0], im.shape[1], 1))
            dsa.append(im)
        imlast = spec[:, -shape:]
        imlast = np.reshape(imlast, (imlast.shape[0], imlast.shape[1], 1))
        dsa.append(imlast)
        return np.array(dsa, dtype=np.float32)

    # Converting from source Spectrogram to target Spectrogram
    def towave(self, spec, name, path='../content/', show=False, ipython=False):
        specarr = self.chopspec(spec)
        print(specarr.shape)
        a = specarr
        print('Generating...')
        ab = self.gen(a, training=False)
        print('Assembling and Converting...')
        a = self.specass(a, spec)
        ab = self.specass(ab, spec)

        IH = Inversion_helpers(self.args)

        awv = IH.deprep(a)
        abwv = IH.deprep(ab)
        print('Saving...')
        pathfin = f'{path}/{name}'
        os.mkdir(pathfin)
        sf.write(pathfin+'/AB.wav', abwv, self.args.sr)
        sf.write(pathfin+'/A.wav', awv, self.args.sr)
        print('Saved WAV!')
        if ipython:
            IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=self.args.sr))
            IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=self.args.sr))
        if show:
            fig, axs = plt.subplots(ncols=2)
            axs[0].imshow(np.flip(a, -2), cmap=None)
            axs[0].axis('off')
            axs[0].set_title('Source')
            axs[1].imshow(np.flip(ab, -2), cmap=None)
            axs[1].axis('off')
            axs[1].set_title('Generated')
            plt.show()
        return abwv
