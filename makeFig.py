#/bin/bash

import matplotlib.pylab as plt
import numpy as np

# Regenerate existing plots?
clobber = False

import os
def make_test():
    file_str = 'figures/test.eps'
    if not clobber and os.path.exists(file_str):
        return

    x = np.linspace(0, 2*np.pi)
    y = np.sin(x)

    fig, axes = plt.subplots(figsize=(8,8))
    axes.plot(x, y)
    plt.savefig(file_str, format='eps', bbox_inches='tight')

make_test()
