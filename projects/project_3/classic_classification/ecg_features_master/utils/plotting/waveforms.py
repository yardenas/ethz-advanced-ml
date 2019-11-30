"""
waveforms.py
--------------------
This module provides functions for plotting ECG waveforms.
--------------------
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import numpy as np
import scipy.io as sio
import matplotlib.pylab as plt
from ipywidgets import interact, fixed


def plot_waveform(raw_data, fs):


    # Get waveform
    time, waveform = np.arange(len(raw_data)) * 1. / fs, raw_data

    # Setup plot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.25)
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # Plot waveform
    ax1.plot(time, waveform, '-k')

    # Configure axes
    ax1.set_xlabel('Time, seconds', fontsize=25)
    ax1.set_ylabel('Amplitude, mV', fontsize=25)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=18)


