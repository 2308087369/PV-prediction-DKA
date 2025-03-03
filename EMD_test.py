import numpy as np
from PyEMD import EMD, Visualisation

t = np.arange(0, 3, 0.01)
S = np.sin(13*t + 0.2*t**1.4) - np.cos(3*t)

# Extract imfs and residue
# In case of EMD
emd = EMD()
emd.emd(S)
imfs, res = emd.get_imfs_and_residue()

# In general:
#components = EEMD()(S)
#imfs, res = components[:-1], components[-1]

vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()