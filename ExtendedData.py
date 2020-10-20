import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall
from scipy.signal import detrend
import BL21BurstData as BL21

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
    '''
        Uses data file info to create an array of frequencies
        Inputs:
            fch1 - Frequency of channel 1
            nchan - Number of channels in data file
            foff - Off frequency range
        Returns:
            Array of frequencies for each frequency channel in data
    '''
    return(fch1 + np.arange(nchan)*foff)

def dedisperse(data, dm, freqs, tsamp):
    '''
        Dedisperses data by input value
        Inputs:
            data - 2D data array
            dm - Desired dispersion measure to dedisperse to
            freqs - Array of frequencies of data channels
            tsamp - Length of time of each time bin
        Returns:
            dedispersed - Dedispersed 2D data
    '''
    delay_bins = []
    dedispersed = []
    for i in range(0, len(freqs)):
        delay_time = 4148808.0 * dm * (1/(freqs[0]**2) - (1/(freqs[i]**2)))/1000
        delay_bins.append(int(np.round(delay_time/tsamp)))
        dedispersed.append(np.zeros(len(data[0]), dtype = np.float32))
    for j in range(0, len(data)):
        dedispersed[j] = np.concatenate([data[j][-delay_bins[j]:], data[j][:-delay_bins[j]]])
    return(dedispersed)

def data_plot(data, name, tag, fax):
    '''
        Makes waterfall plot of input data
        Inputs:
            data - Array of data arrays
            name - save name
            tag - save tag
        Returns:
            nothing
    '''
    plt.imshow(data, origin = 'lower', interpolation = 'nearest', aspect = 'auto', vmin = 0, vmax = 170*20, extent = [0, 200, fax[0], fax[len(fax)-1]])
    cbar = plt.colorbar()
    cbar.set_label('Flux Density')
    plt.ylabel('Frequency (MHz)')
    plt.xlabel('Time Bins')
    plt.title(name + ' Data of Burst ' + tag)
    plt.savefig(name + '_' + tag)

def fscrunch(data, freqs, nchan, factor):
    newnchan = nchan//factor
    newfreq = np.zeros(newnchan)
    for k in range(newnchan):
        newfreq[k] = np.sum(freqs[k*factor:(k+1)*factor])/float(factor)
    retval = np.zeros((len(np.linspace(0, nchan, len(newfreq))), len(data[0])))
    for k in range(newnchan):
        for l in range(len(data[0])):
            tot = 0
            for i in range(k*factor, (k+1)*factor):
                tot += data[i][l]
            retval[k][l] = tot
    return(retval, newfreq)

def fluence_data(namefile):
    read_data = open(namefile, 'r')
    bursts = []
    while True:
        line = read_data.readline()
        if not line:
            break
        splitline = line.split()
        if splitline[0][0] == '#':
            pass
        else:
            tag = str(splitline[0])
            filename = str(splitline[1])
            tsamp = int(splitline[2])
            bursts.append([tag, filename, tsamp])
    freqs = get_freqs(fch1 = 8161.132568359375, nchan = 10924, foff = -0.3662109375)    #nchan = 14848? fch1 = 9313.78173828125?
    for i in range(0, len(bursts)):
        data = load(filename = bursts[i][1], info = False, tstart = bursts[i][2]-200, tstop = bursts[i][2]+200)
        dedisdata = dedisperse(data = data, dm = 558, freqs = freqs, tsamp = 0.0003495253333333333)
        scrunchdat, fax = fscrunch(data = dedisdata[:10880], freqs = freqs[:10880], nchan = 10880, factor = 170)     #Original data has nchan = 10924
        #peak = BL21.find_peak(data = scrunchdat)
        #params = BL21.comp_param(data = scrunchdat, mode = 'gaussian', n = 1, pllim = [50, 105, 0, 0], phlim = [100, 0, 0, 0], fllim = [5, 0, 0, 0], fhlim = [40, 0, 0, 0], factor = 78.3, fax = fax, tag = bursts[i][0])
        #BL21.comp_plot(data = [params[3][0]], name = 'Fluence', fax = fax, units = 'Jy ms', tag = 'FB' + bursts[i][0], labels = ('F'), log = False, RSN = False)
        #BL21.fit(burst = scrunchdat[14], mode = 'gaussian', n = 1, llimit = 50, hlimit = 100, freq = fax[14], tag = '11A', plot = True)
        data_plot(data = scrunchdat, name = '121102-Filterbank', tag = bursts[i][0], fax = fax)
        plt.clf()

def main():
    fluence_data(namefile = 'full_data.txt')

main()
