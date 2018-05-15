"""
Here we are trying to learn how to fit a model to our data.

Simply running the code will generate some helpful plots and print out some information to the console.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve

# Generate a binary stochastic signal of the length `length`
def generate_stochastic_signal(length):
    signal = np.resize([1, 0], length)
    for i in range(1, length):
        if random.random() > 0.995:
            signal[i] = -1 * signal[i-1]
        else:
            signal[i] = signal[i-1]
    return signal.tolist()

sig = generate_stochastic_signal(8192*4)
win = json.load(open('./data/transducer_id/system.sys', 'r'))
filtered = convolve(sig, win, mode='full')[:len(sig)]

fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.set_title('Original pulse')
ax_orig.margins(0, 0.1)
ax_win.plot(win)
ax_win.set_title('Filter impulse response')
ax_win.margins(0, 0.1)
ax_filt.plot(filtered)
ax_filt.set_title('Filtered signal')
ax_filt.margins(0, 0.1)
fig.tight_layout()
plt.show()
