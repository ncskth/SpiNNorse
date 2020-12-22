import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import time
import os

def fetch_files(path):

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if ('spikes_' in file) and ('.csv' in file):
                files.append(os.path.join(r, file))

    files = sorted([f.lower() for f in files])

    return files
    
def plot_in_v_out(i_indexes, v, o_indexes, sfn, xlim):

    # Plot spikes and voltages
    fig, axs = plt.subplots(3, figsize=(15,9))
    fig.tight_layout(pad=5.0)
    fig.suptitle(sfn)

    axs[0].eventplot(i_indexes, linewidths=2, colors='k') # Plot the timesteps where the neuron spiked
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Input")
    axs[0].set_xlim((0,xlim))
    axs[0].grid()

    axs[1].plot(v, linewidth=2, color='g')
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Voltage [ms]")
    axs[1].set_xlim((0,xlim))
    axs[1].grid()

    axs[2].eventplot(o_indexes, linewidths=2, colors='g') # Plot the timesteps where the neuron spiked
    axs[2].set_xlabel("Time [ms]")
    axs[2].set_ylabel("Output")
    axs[2].set_xlim((0,xlim))
    axs[2].grid()
