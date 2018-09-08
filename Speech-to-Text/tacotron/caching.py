import tqdm
import os
import numpy as np
from setting import *

if not os.path.exists("spectrogram"): os.mkdir("spectrogram")
fpaths = os.listdir(data+'wavs')
for fpath in tqdm.tqdm(fpaths):
    fname, spectrogram = load_file('%swavs/%s'%(data,fpath))
    np.save("spectrogram/{}".format(fname.replace("wav", "npy")), spectrogram)
