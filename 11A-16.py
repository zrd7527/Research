import pypulse as p
import matplotlib.pyplot as plt
import numpy as np
import pypulse.utils as u
import pypulse.rfimitigator as rfim

def start(filename):
    '''
        Creates archive object, performs basic noise mitigation, and averages bins and channels to reasonable sizes
        Inputs:
            filename - File to be opened
        Returns:
            ar - archive object
            data - Array of data arrays, with each data array being at a specific frequency channel
            fax - Frequency axis, corresponds to the arrays in data
    '''
    ar = p.Archive(filename)
    ar.dedisperse(reverse = True)
    rm = mitigate(ar)
    nulows = [7800]
    nuhighs = [8400]
    zap_freq(rm, nulows = nulows, nuhighs = nuhighs)
    ar.bscrunch(nbins = 2048, factor = 4)       # Average Burst in Phase
    ar.fscrunch(nchan = 19456, factor = 304)    # Average Burst in Frequency
    data = ar.getData()
    fax = ar.getAxis(flag = 'F')        # Frequency Axis
    return(ar, data, fax)

def find_peak(data):
    '''
        Finds peak flux value and data array at the peak flux frequency 
        Inputs:
            data - Array of data arrays
        Outputs:
            peak - Peak flux value
            burst - Data array containing peak flux value
            freq_index - Array index of input data containing peak value
    '''
    peak = 0        # Peak Value
    burst = []      # Burst Data Array
    freq = 0        # Frequency Index
    for i in range(0,len(data)):          # Move Through Frequency Channels
        for j in range(0,len(data[i])):   # Move Through Phase Bins
            if data[i][j] > peak:
                peak = data[i][j]
                burst = data[i]
                freq_index = i
    return(peak, burst, freq_index)

def burst_prop(burst):
    '''
        Finds burst properties of input data array
        Inputs:
            burst - Data array
        Retrns:
            sp - singlepulse object
            FWHM - Full width at half max of input data array
            SN - Signal to noise of input data array
    '''
    sp = p.SinglePulse(data = burst, windowsize = 256)
    FWHM = sp.getFWHM()
    SN = sp.getSN()
    return(sp, FWHM, SN)

def freq_av(data):
    ''' 
        Simple average in frequency
        Inputs:
            data - Array of data arrays
        Returns:
            av - Array of flux values of burst averaged in frequency
    '''
    tot = []
    count = 1
    for i in range(0, len(data)):
        if i == 0:
            tot = data[i]
        else:
            tot += data[i]
            count += 1
    av = tot/count
    return(av)

def mitigate(ar):
    ''' 
        Finds and removes dead channels and rfi
        Inputs:
            ar - archive object
        Returns:
            rm - rfimitigator object
    '''
    rm = rfim.RFIMitigator(ar)
    rm.zap_minmax()             #Auto-Zap Dead Channels
    return(rm)

def zap_freq(rm, nulows, nuhighs):
    ''' 
        Removes dead channels between input frequency ranges, which must be in order
        Inputs:
            rm - rfimitigator object
            nulows - Array of lower bounds of frequency ranges to zap
            nuhighs - Array of upper bounds of frequency ranges to zap
        Returns:
            rm - Modified rfimitigator object
    '''
    if len(nulows) != len(nuhighs):     #Check for Valid Ranges
        return()
    for i in range(0, len(nulows)):
        rm.zap_frequency_range(nulow = nulows[i], nuhigh = nuhighs[i])
    return(rm)

def unweight(ar, frequencies):
    ''' 
        Sets statistical weight of input frequencies to 0
        Inputs:
            ar - archive object
            frequencies - Array of frequencies to be weighted to 0
        Returns:
            ar - Modified archive object
    '''
    ar.setWeights(val = 0.0, f = frequencies)
    return(ar)

def destroy_greater(data, index):
    ''' 
        Removes all frequencies greater than input - bad practice, use only for simple analyses of extremely noisy bursts
        Inputs:
            data - Array of data arrays
            index - Lower bound of deleted frequencies
        Returns:
            data - Modified array of data arrays
    '''
    for i in range(index, len(data)):
        for j in range(0, len(data[i])):
            data[i][j] = 0.0
    return(data)

def destroy_lower(data, index):
    ''' 
        Removes all frequencies lower than input - bad practice, use only for simple analyses of extremely noisy bursts
        Inputs:
            data - Array of data arrays
            index - Upper bound of deleted frequencies
        Returns:
            data - Modified array of data arrays
    '''
    for i in range(0, index):
        for j in range(0, len(data[i])):
            data[i][j] = 0.0
    return(data)

def comp_param(data, mode, n, llim, hlim, mind, mamp, mmu, mwidth, fax, tag):
    ''' In Development
        Finds parameters of data components
        Inputs:
            data - Array of data arrays
            mode - Desired type of fit, gaussian or vonmises
            n - number of components to be fit
            llim - Array of lower bounds of components
            hlim - Array of upper bounds of components
            mind - Manually entered index range, first element is low index second is high limit
            mamp - Manually entered amplitudes for poor fits, array of arrays
            mmu - Manually entered centers for poor fits, array of arrays
            mwidth - Manually entered widths for poor fits, array of arrays
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
        Returns:
            amps - Array of amplitude arrays, organized into separate components
            mus - Array of center arrays, organized into separate components
            widths - (not yet included) Array of width arrays, organized into separate components
    '''
    amps = [ [], [], [], [] ]
    mus = [ [], [], [], [] ]
    widths = [ [], [], [], [] ]
    for i in range(0, len(data)):
        if i in range(mind[0], mind[1]):
            amps[0].append(mamp[i-mind[0]][0])
            amps[1].append(mamp[i-mind[0]][1])
            amps[2].append(mamp[i-mind[0]][2])
            amps[3].append(mamp[i-mind[0]][3])
            mus[0].append(mmu[i-mind[0]][0])
            mus[1].append(mmu[i-mind[0]][1])
            mus[2].append(mmu[i-mind[0]][2])
            mus[3].append(mmu[i-mind[0]][3])
            widths[0].append(mwidth[i-mind[0]][0])
            widths[1].append(mwidth[i-mind[0]][1])
            widths[2].append(mwidth[i-mind[0]][2])
            widths[3].append(mwidth[i-mind[0]][3])
        else:
            GetFit = fit(burst = data[i], mode = mode, n = n, freq = fax[i], tag = tag, plot = False)
            for j in range(0, len(GetFit[1])):
                if llim[0] < GetFit[1][j] < hlim[0]:
                    x = burst_prop(data[i][llim[0]:llim[1]])
                    widths[0].append(x[1])
                    amps[0].append(GetFit[0][j])
                    mus[0].append(GetFit[1][j])
                elif llim[1] < GetFit[1][j] < hlim[1]:
                    if np.sum(data[i][(hlim[0]+1):llim[2]]) != 0:
                        x = burst_prop(data[i][(hlim[0]+1):llim[2]])
                        widths[1].append(x[1])
                    amps[1].append(GetFit[0][j])
                    mus[1].append(GetFit[1][j])
                elif llim[2] < GetFit[1][j] < hlim[2]:
                    x = burst_prop(data[i][(hlim[1]+1):llim[3]])
                    widths[2].append(x[1])
                    amps[2].append(GetFit[0][j])
                    mus[2].append(GetFit[1][j])
                elif llim[3] < GetFit[1][j] < hlim[3]:
                    x = burst_prop(data[i][(hlim[2]+1):(hlim[3]+1)])
                    widths[3].append(x[1])
                    amps[3].append(GetFit[0][j])
                    mus[3].append(GetFit[1][j])
            if (len(mus[0]) - 1) < i:
                mus[0].append(np.nan)
                amps[0].append(0)
                widths[0].append(0)
            if (len(mus[1]) - 1) < i:
                mus[1].append(np.nan)
                amps[1].append(0)
                widths[1].append(0)
            if (len(mus[2]) - 1) < i:
                mus[2].append(np.nan)
                amps[2].append(0)
                widths[2].append(0)
            if (len(mus[3]) - 1) < i:
                mus[3].append(np.nan)
                amps[3].append(0)
                widths[3].append(0)
    return(amps, mus, widths)

def manual_gaussians(x, amp, mu, sigma):
    '''
        Makes Gaussian curves with input parameters
        Inputs:
            x - x axis
            amp - Amplitudes of curves, array
            mu - Centers of curves, array
            sigma - Standard deviations of curves, array
        Returns:
            curve - Array of Gaussian curve values
    '''
    curve = np.zeros(len(x))
    for i in range(0, len(amp)):
        curve += u.gaussian(x = x, amp = amp[i], mu = mu[i], sigma = sigma[i])
    return(curve)

def fit(burst, mode, n, freq, tag, plot):
    '''
        Fits n components to data array and can plot fit and burst for comparison
        (Does not always fit properly due to significance test)
        Inputs:
            burst - Data array to be fit
            mode - Desired type of fit, can be gaussian or vonmises
            n - Number of components to be fit
            freq - Frequency of data array, for plotting
            tag - Burst name, e.g. 11A, for plotting
            plot - Boolean, true to make plot
        Returns:
            amp - Array of component amplitudes
            mu - Array of component centers
    '''
    x = np.linspace(start=1, stop=512, num=512)
    amp = []
    mu = []
    ForceFit = u.fit_components(xdata = x, ydata = burst, mode = mode,  N=n)
    pfit = ForceFit[2]
    retval = np.zeros(len(x))
    for j in range(n):
        retval += u.gaussian(x, pfit[3*j], pfit[3*j+1], pfit[3*j+2])
        amp.append(pfit[3*j])
        mu.append(pfit[3*j+1])
    if plot == True:
        plt.plot(x, retval, 'k')
        plt.plot(x, burst)
        plt.xlim(340,430)          #Zoom in on data
        plt.xlabel('Phase Bins')
        plt.ylabel('Flux Density')
        plt.title(tag + ' Peak Flux (at ' + str(round(freq)) + ' MHz)')
        plt.savefig(tag + '_Fit_Test')
    return(amp, mu)

def data_plot(data, fax, tag):
    ''' 
        Makes 3D data plot of entire data file, x is phase, y is frequency, z is flux density
        Inputs:
            data - Array of data arrays
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
        Returns:
            nothing
    '''
    plt.imshow(X = data, aspect = 'auto', interpolation = 'nearest', origin = 'lower', extent = [0,512,fax[0],fax[len(fax)-1]])
    plt.xlabel('Phase Bins')
    plt.ylabel('Frequency(MHz)')
    plt.title('Burst ' + tag + ', Dead Channels Removed')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density')
    plt.savefig(tag + '_Data')
    cbar.remove()

def comp_plot(data, name, fax, tag, labels, log):
    ''' 
        Makes 2D plot of component parameters vs frequency
        Inputs:
            data - Array of component parameters
            name - Name of parameter to be plotted, e.g. Amplitude
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
            labels - Component name list for legend
            log - Boolean, true for log plot
        Returns:
            nothing
    '''
    for i in data:
        plt.plot(fax, i)
    plt.legend(labels = labels)
    if log == True:
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Log Frequency')
        plt.ylabel('Log ' + name)
    else:
        plt.xlabel('Frequency')
        plt.ylabel(name)
    plt.title(name + ' Versus Frequency of Components of Burst ' + tag)
    plt.savefig(tag + '_' + name)

def main():
    new = start(filename = '11A_16sec.calib.4p')
    '''
    tag = '11A'
    LowLims = [350, 363, 380, 395]
    HighLims = [362, 370, 390, 420]
    MIndices = [17, 22]
    MAmp = [[4000, 3500, 0, 0], [8000, 11000, 0, 0], [14000, 18000, 3000, 0], [15000, 25500, 5000, 0], [5500, 12000, 2500, 0]]
    MMu = [[358, 365.5, np.nan, np.nan], [359, 366, np.nan, np.nan], [360, 366, 382, np.nan], [360, 366, 382, np.nan], [360, 366.5, 385, np.nan]]
    MWidth = [[2, 2, 0, 0], [3, 1, 0, 0], [3, 1, 1, 0], [3, 1, 1, 0], [3, 1, 5, 0]]
    labels = ('Comp 1', 'Comp 2', 'Comp 3', 'Comp 4')
    params = comp_param(data = new[1], mode = 'gaussian', n = 4, llim = LowLims, hlim = HighLims, mind = MIndices, mamp = MAmp, mmu = MMu, mwidth = MWidth, fax = new[2], tag = tag)
    comp_plot(data = params[2], name = 'Width_Test', fax = new[2], tag = tag, labels = labels, log = False)
    '''
    x = find_peak(data = new[1])
    print(x[0])

main()
