import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall
from scipy.signal import detrend

def load(filename, info, tstart, tstop):
    '''
        Loads file data as a waterfall plot type array and can print data info
        Inputs:
            filename - name of file being loaded
            info - Boolean, True to print data info
            tstart - Starting time bin
            tstop - Ending time bin
        Returns:
            cleandata - Detrended (bandpass removed) array of data arrays
    '''
    fb = Waterfall(filename, t_start = tstart, t_stop = tstop)
    if info == True:
        fb.info()
    freqs, data = fb.grab_data(4000, 8000, 1)
    newdat = []
    for i in range(0, len(data[0])):
        newarr = []
        for j in range(0, len(data)):
            newarr.append(data[j][i])
        newdat.append(newarr)
    cleandata = detrend(newdat)
    return(cleandata)

def data_plot(data, name, tag):
    '''
        Makes waterfall plot of input data
        Inputs:
            data - Array of data arrays
            name - save name
            tag - save tag
        Returns:
            nothing
    '''
    plt.imshow(data, origin = 'lower', interpolation = 'nearest', aspect = 'auto')
    plt.ylabel('Frequency Bins')
    plt.xlabel('Time Bins')
    plt.savefig(name + '_' + tag)

def main():
    data1 = load(filename = 'spliced_guppi_57991_51723_DIAG_FRB121102_0012.gpuspec.0001.8.fil', info = False, tstart = 894000, tstop = 895000)
    data_plot(data = data1, name = 'BurstSearch', tag = '51723')
main()
