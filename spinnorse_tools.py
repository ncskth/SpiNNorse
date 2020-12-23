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

def plot_iho(i_spikes, o_spikes_hl, v_hl, o_spikes_ol, v_ol, sfn, xlim, ratio):

    # Set figure size 
    nsll = 0.4  # neuron-spike line-length
    extra = 0
    
    rv = 4    
    ri = (i_spikes.shape[1]+2)*(nsll)+extra
    rh = (o_spikes_hl.shape[1]+2)*(nsll)+extra
    ro = (o_spikes_ol.shape[1]+2)*(nsll)+extra
    
    height = (ri+rv+rh+rv+ro)+2
    width = 15
    
    
    fig, axs = plt.subplots(5, figsize=(width,height), gridspec_kw={'height_ratios': [ri, rv, rh, rv, ro]})
    fig.tight_layout(pad=5.0)
    fig.suptitle(sfn)

    n = i_spikes.shape[1]
    for i in range(n):
        i_indexes = np.where(i_spikes[:,i]>0)
        axs[0].eventplot(i_indexes, linewidths=2, colors='k', lineoffsets=i+1, linelengths=nsll) 
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Spikes in IL")
    axs[0].set_xlim((0,xlim))
    axs[0].set_ylim((0,n+1))
    axs[0].set_xticks(np.arange(0, xlim+1, 10))
    axs[0].set_yticks(np.arange(0, n+2, 1))
        
    n = v_hl.shape[1]
    for i in range(n):
        c = 'C{}'.format(i)
        axs[1].plot(v_hl[:,i], linewidth=2)
    axs[1].set_title("Hidden Layer")
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Voltage [ms]")
    axs[1].set_xlim((0,xlim))

    n = o_spikes_hl.shape[1]
    for i in range(n):
        o_indexes = np.where(o_spikes_hl[:,i]>0)
        axs[2].eventplot(o_indexes, linewidths=2, colors='k', lineoffsets=i+1, linelengths=nsll) 
    axs[2].set_xlabel("Time [ms]")
    axs[2].set_ylabel("Spikes in HL")
    axs[2].set_xlim((0,xlim))
    axs[2].set_ylim((0,n+1))
    axs[2].set_xticks(np.arange(0, xlim+1, 10))
    axs[2].set_yticks(np.arange(0, n+2, 1))
                
    n = v_ol.shape[1]
    for i in range(n):
        c = 'C{}'.format(i)
        axs[3].plot(v_ol[:,i], linewidth=2)
    axs[3].set_title("Output Layer")
    axs[3].set_xlabel("Time [ms]")
    axs[3].set_ylabel("Voltage [ms]")
    axs[3].set_xlim((0,xlim))

    n = o_spikes_ol.shape[1]
    for i in range(n):
        o_indexes = np.where(o_spikes_ol[:,i]>0)
        axs[4].eventplot(o_indexes, linewidths=2, colors='k', lineoffsets=i+1, linelengths=nsll) 
    axs[4].set_xlabel("Time [ms]")
    axs[4].set_ylabel("Spikes in OL")
    axs[4].set_xlim((0,xlim))
    axs[4].set_ylim((0,n+1))
    axs[4].set_xticks(np.arange(0, xlim+1, 10))
    axs[4].set_yticks(np.arange(0, n+2, 1))