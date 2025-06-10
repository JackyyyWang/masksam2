import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
spacing_list = []
for file_ in glob('/home/z005257c/Documents/nnUNet_preprocessed/Dataset003_PROSTATE/nnUNetPlans_3d_fullres/*.pkl'):
    xx = np.load(file_, allow_pickle=True)['spacing'][0]
    spacing_list.append(xx)
yy = plt.hist(spacing_list)
plt.show()
print(yy)