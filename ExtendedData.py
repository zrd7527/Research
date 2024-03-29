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

def data_plot(data, name, tag, fax, vmax, ext):
    '''
        Makes waterfall plot of input data
        Inputs:
            data - Array of data arrays
            name - Save name
            tag - Save tag
            fax - Array of frequency axis values
            vmax - Maximum data plot value (max colorbar value as well)
            ext - Maximum extent of time bins on x-axis
        Returns:
            nothing
    '''
    TimeConversion = 25.6
    FluxConversion = 64.5
    newdat = []
    for i in range(len(data)):
        newdat.append(data[i]/FluxConversion)
    plt.imshow(newdat, origin = 'upper', interpolation = 'nearest', aspect = 'auto', vmin = 0, vmax = vmax/FluxConversion, extent = [0, ext/TimeConversion, fax[len(fax)-1], fax[0]])
    cbar = plt.colorbar()
    cbar.set_label('Flux Density (mJy)')
    plt.ylabel('Frequency (MHz)')
    plt.xlabel('Time (ms)')
    plt.title(name + ' Data of Burst ' + tag)
    plt.savefig(name + '_' + tag)

def fscrunch(data, freqs, nchan, factor):
    '''
    Scrunches data along frequency axis to average data down for more visible plotting and analysis
    Inputs:
        data - Array of data arrays
        freqs - Frequency axis array
        nchan - Original number of frequency channels
        factor - Number to divide nchan by to average over frequency
    Returns:
        retval - Averaged data, array of arrays
        newfreq - New frequency axis array
    '''
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

def bscrunch(data, nbins, factor):
    '''
    Scrunches data along time axis similarly to fscrunch()
    Inputs:
        data - Array of data arrays
        nbins - Number of time bins in each data array
        factor - Number to divide nbins by
    Returns:
        retval - Data averaged along time dimension
    '''
    newnbins = nbins//factor
    retval = np.zeros(shape = (len(data), len(np.arange(start = 0, stop = nbins, step = factor))))
    counts = np.zeros_like(retval)
    for i in range(factor):
        arr = data[:, i:nbins:factor]
        count = np.ones_like(arr)
        length = np.shape(arr)[1]
        retval[:, :length] += arr
        counts[:, :length] += count
    retval = retval/counts
    return(retval)

def extract_bursts(namefile, plot):
    '''
    Uses input text file of burst file and time locations to pull out bursts, average in frequency and time if necessary,
    plot if desired, then return the data array
    Inputs:
        namefile - Text file containing columns of burst tags, data file, TOA in time bins from start of file, and DM
                   Also includes peak frequency and width of burst if needed for fitting
        plot - Boolean, True to plot the cleaned bursts
    Returns:
        bursts - Array of all bursts in data set. Each element is an Array with all information of that burst.
    '''
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
            DM = float(splitline[3])
            width = float(splitline[4])
            nupeak = float(splitline[5])
            bursts.append([tag, filename, tsamp, DM, width, nupeak])
    freqs = get_freqs(fch1 = 8161.132568359375, nchan = 10924, foff = -0.3662109375)    #nchan = 14848? fch1 = 9313.78173828125? from file info
    for i in range(0, len(bursts)):
        data = load(filename = bursts[i][1], info = False, tstart = bursts[i][2]-200, tstop = bursts[i][2]+200)
        ext = 400
        dedisdata = dedisperse(data = data, dm = bursts[i][3], freqs = freqs, tsamp = 0.0003495253333333333)
        if len(bursts[i][0]) > 3:   #Naming convention for low S/N bursts in first and second file
            fscrunchdat, fax = fscrunch(data = dedisdata[:10912], freqs = freqs[:10912], nchan = 10912, factor = 682)
            scrunchdat = bscrunch(data = fscrunchdat, nbins = ext, factor = 4)
            best_vmax = 170*8
        elif int(bursts[i][0][0:2]) > 12:   #Naming convention for bursts after second file
            fscrunchdat, fax = fscrunch(data = dedisdata[:10912], freqs = freqs[:10912], nchan = 10912, factor = 682)
            scrunchdat = bscrunch(data = fscrunchdat, nbins = ext, factor = 4)
            best_vmax = 170*8
        else:
            scrunchdat, fax = fscrunch(data = dedisdata[:10880], freqs = freqs[:10880], nchan = 10880, factor = 170)     #Original data has nchan = 10924
            best_vmax = 170*20
        bursts[i].append(fax)
        bursts[i].append(scrunchdat)
        if plot == True:
            data_plot(data = scrunchdat, name = '121102-Filterbank', tag = bursts[i][0], fax = fax, vmax = best_vmax, ext = ext)
        plt.clf()
    return(bursts)

def get_fluence(bursts, plot_center):
    '''
    Takes Input array of burst data arrays and uses BL21BurstData.py file to find fluence, width, amplitude, and center of each burst
    Inputs:
        bursts - Array of arrays; each array element contains all information for that burst
        plot_center - Boolean, True to plot data with overlayed gaussian center
    Reurns:
        None
    '''
    tfdarr = []
    for i in range(len(bursts)):
        if len(bursts[i][0]) < 4:
            pass
        else:
            tag = bursts[i][0]
            tfdarr.append([tag])
            print(tag)
            tsamp = bursts[i][2]
            nupeakGHz = bursts[i][5]
            fax = bursts[i][6]
            data = bursts[i][7]
            peak, burst, nupeakind, tbin = BL21.find_peak(data)
            pllim = [tbin-5, tbin+6, 0, 0]
            phlim = [tbin+5, 0, 0, 0]
            if tag == "11B2":
                fllim = [nupeakind-4, 0, 0, 0]
                fhlim = [nupeakind+4, 0, 0, 0]
            else:
                fllim = [nupeakind-2, 0, 0, 0]
                fhlim = [nupeakind+2, 0, 0, 0]
            try:
                params = BL21.comp_param(data = data, mode = 'gaussian', n = 1, pllim = pllim, phlim = phlim, fllim = fllim, fhlim = fhlim, factor = 20.5, fax = fax, tag = tag)
                tfdarr[-1].append([params[3][0]])
                tfdarr[-1].append(data)
                if plot_center == True:
                    BL21.data_plot(data = scrunchdat, tag = bursts[i][0], fax = fax, center = params[1], RSN = False, vmax = 170*8, ext = ext)
                plt.clf()
            except ValueError:
                print("No fit found for burst " + str(tag))
                tfdarr.pop(-1)
            except IndexError:
                print("Burst " + str(tag) + " not properly fit")
                tfdarr.pop(-1)
            '''
            if tag == "11A1":
                BL21.comp_plot(data = [params[3][0]], name = 'Fluence', fax = fax, units = 'Jy ms', tag = 'FB' + bursts[i][0], labels = ('F'), log = False, RSN = False)
            elif tag == "11B2":
                BL21.comp_plot(data = [params[3][0]], name = 'Fluence', fax = fax, units = 'Jy ms', tag = 'FB' + bursts[i][0], labels = ('F'), log = False, RSN = False)
            '''
    return(tfdarr)

def fluence_moment_scatt(moment, RSN, singleA):
    '''
    This function plot the total fluence vs. statistical moments of original 21 bursts
    and an example few from the extended data set
    '''
    #First get info from original data set (4p files)
    single_comp_BL21_tfdmarr = BL21.burst_stats(multi = False, plot = False)[3] #single component burst info
    multi_comp_BL21_tfdmarr = BL21.burst_stats(multi = True, plot = False)[3] #multi component burst info
    for i in range(len(multi_comp_BL21_tfdmarr)):
        if multi_comp_BL21_tfdmarr[i][0] == '11E':
            pass
        elif multi_comp_BL21_tfdmarr[i][0] == '11K':
            pass
        elif multi_comp_BL21_tfdmarr[i][0] == '11O':
            pass
        else:
            single_comp_BL21_tfdmarr.append(multi_comp_BL21_tfdmarr[i])
    combined_tfdmarr = single_comp_BL21_tfdmarr
    #Now find the info for extended data set bursts (filterbank files)
    BurstInfo = extract_bursts(namefile = 'full_data.txt', plot = False)
    tfdarr = get_fluence(bursts = BurstInfo, plot_center = False)
    for i in range(len(tfdarr)):
        moms = BL21.moments(tfdarr[i][1])
        tfdarr[i].append(moms)
        combined_tfdmarr.append(tfdarr[i])
    BL21.fluence_moment_scatt(tfdmarr = combined_tfdmarr, moment = moment, RSN = RSN, singleA = singleA)

def main():
    files = ["spliced_guppi_57991_49905_DIAG_FRB121102_0011.gpuspec.0001.8.fil", "spliced_guppi_57991_51723_DIAG_FRB121102_0012.gpuspec.0001.8.fil", "spliced_guppi_57991_53535_DIAG_FRB121102_0013.gpuspec.0001.8.fil", "spliced_guppi_57991_55354_DIAG_FRB121102_0014.gpuspec.0001.8.fil", "spliced_guppi_57991_57166_DIAG_FRB121102_0015.gpuspec.0001.8.fil", "spliced_guppi_57991_58976_DIAG_FRB121102_0016.gpuspec.0001.8.fil", "spliced_guppi_57991_60787_DIAG_FRB121102_0017.gpuspec.0001.8.fil", "spliced_guppi_57991_62598_DIAG_FRB121102_0018.gpuspec.0001.8.fil", "spliced_guppi_57991_64409_DIAG_FRB121102_0019.gpuspec.0001.8.fil", "spliced_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.8.fil"]
    for i in range(len(files)):
        dat = load(filename = files[i], info = False, tstart = 0, tstop = 1000)
        BurstInfo = extract_bursts(namefile = 'full_data.txt', plot = True)
    #print(get_fluence(bursts = BurstInfo, plot_center = False))
    #fluence_moment_scatt(moment = 'Skew', RSN = False, singleA = False)

main()
