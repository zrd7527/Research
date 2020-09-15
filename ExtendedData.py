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

def get_freqs(fch1, nchan, foff):
    return(fch1 + np.arange(nchan)*foff)

def dedisperse(data, dm, freqs, tsamp):
    delay_bins = []
    dedispersed = []
    for i in range(0, len(freqs)):
        delay_time = 4148808.0 * dm * (1/(freqs[0]**2) - (1/(freqs[i]**2)))/1000
        delay_bins.append(int(np.round(delay_time/tsamp)))
        dedispersed.append(np.zeros(len(data[0]), dtype = np.float32))
    for j in range(0, len(data)):
        dedispersed[j] = np.concatenate([data[j][-delay_bins[j]:], data[j][:-delay_bins[j]]])
    return(dedispersed)

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
    data1 = load(filename = 'spliced_guppi_57991_49905_DIAG_FRB121102_0011.gpuspec.0001.8.fil', info = False, tstart = 46000, tstop = 47000)
    freqs = get_freqs(fch1 = 9313.78173828125, nchan = 14848, foff = -0.3662109375)
    dedisdata = dedisperse(data = data1, dm = 558, freqs = freqs, tsamp = 0.0003495253333333333)
    data_plot(data = dedisdata, name = 'Dedispersed', tag = '11A')
main()
