import tqdm
import os
import numpy as np
from utils import *

if not os.path.exists("mel"): os.mkdir("mel")
if not os.path.exists("mag"): os.mkdir("mag")
fpaths = os.listdir(data+'wavs')
for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_file('%swavs/%s'%(data,fpath))
    np.save("mel/{}".format(fname.replace("wav", "npy")), mel)
    np.save("mag/{}".format(fname.replace("wav", "npy")), mag)
