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
    rm.zap_minmax()             # Auto-Zap Dead Channels
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
    if len(nulows) != len(nuhighs):     # Check for Valid Ranges
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

def comp_param(data, mode, n, pllim, phlim, fllim, fhlim, fax, tag):
    ''' 
        Finds parameters of data components
        Inputs:
            data - Array of data arrays
            mode - Desired type of fit, gaussian or vonmises
            n - number of components to be fit
            pllim - Array of lower phase bounds of components
            phlim - Array of upper phase bounds of components
            fllim - Array of lower frequency bounds of components
            fhlim - Array of upper frequency bounds of components
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
        Returns:
            amps - Array of amplitudes for each component
            mus - Array of centers for each component
            widths - Array of widths for each component
            fluence - Array of fluences for each component
    '''
    amps = [ [], [], [], [] ]                   # Initialize component arrays, can handle max 4 components
    mus = [ [], [], [], [] ]
    widths = [ [], [], [], [] ]
    fluence = [[], [], [], []]
    for i in range(0, len(data)):
        GetFit = fit(burst = data[i], mode = mode, n = n, llimit = pllim[0], hlimit = phlim[n-1], freq = fax[i], tag = tag, plot = False)   # Automatic fit routine
        for j in range(0, len(GetFit[1])):
            if pllim[0] < GetFit[1][j] < phlim[0] and (len(mus[0]) - 1) < i and fllim[0] < i < fhlim[0]:         # Check if component center is within given phase limits and frequency limits
                x = burst_prop(data[i][(pllim[0]-4):pllim[1]])
                widths[0].append(x[1])
                amps[0].append(GetFit[0][j])
                mus[0].append(GetFit[1][j])
                fluence[0].append(np.sum(GetFit[2][0:(phlim[0]-pllim[0])])/(1000*2892))     # Sum area under fit curve then convert from flux density and phase bin units to Jy*ms
            elif pllim[1] < GetFit[1][j] < phlim[1] and (len(mus[1]) - 1) < i and fllim[1] < i < fhlim[1]:
                x = burst_prop(data[i][(phlim[0]+1):pllim[2]])    # Find FWHM using previous component high limit and next component low limit\
                widths[1].append(x[1])                          # Giving extra noise around component provides more accurate FWHM
                amps[1].append(GetFit[0][j])
                mus[1].append(GetFit[1][j])
                fluence[1].append(np.sum(GetFit[2][(pllim[1]-pllim[0]):(phlim[1]-pllim[0])])/(1000*2892))
            elif pllim[2] < GetFit[1][j] < phlim[2] and (len(mus[2]) - 1) < i and fllim[2] < i < fhlim[2]:
                x = burst_prop(data[i][(phlim[1]+1):pllim[3]])
                widths[2].append(x[1])
                amps[2].append(GetFit[0][j])
                mus[2].append(GetFit[1][j])
                fluence[2].append(np.sum(GetFit[2][(pllim[2]-pllim[0]):(phlim[2]-pllim[0])])/(1000*2892))
            elif pllim[3] < GetFit[1][j] < phlim[3] and (len(mus[3]) - 1) < i and fllim[3] < i < fhlim[3]:
                x = burst_prop(data[i][(phlim[2]+1):(phlim[3]+1)])
                widths[3].append(x[1])
                amps[3].append(GetFit[0][j])
                mus[3].append(GetFit[1][j])
                fluence[3].append(np.sum(GetFit[2][(pllim[3]-pllim[0]):(phlim[3]-pllim[0])])/(1000*2892))
        if (len(mus[0]) - 1) < i:                   # For the case of no component found
            mus[0].append(np.nan)
            amps[0].append(0)
            widths[0].append(0)
            fluence[0].append(0)
        if (len(mus[1]) - 1) < i:
            mus[1].append(np.nan)
            amps[1].append(0)
            widths[1].append(0)
            fluence[1].append(0)
        if (len(mus[2]) - 1) < i:
            mus[2].append(np.nan)
            amps[2].append(0)
            widths[2].append(0)
            fluence[2].append(0)
        if (len(mus[3]) - 1) < i:
            mus[3].append(np.nan)
            amps[3].append(0)
            widths[3].append(0)
            fluence[3].append(0)
    return(amps, mus, widths, fluence)

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
        if amp[i] > 0:                  # To handle arrays with less than max components so nan does not get passed in
            curve += u.gaussian(x = x, amp = amp[i], mu = mu[i], sigma = sigma[i])
    return(curve)

def fit(burst, mode, n, llimit, hlimit, freq, tag, plot):
    '''
        Fits n components to data array and can plot fit with burst for comparison
        Only fits properly when given a small phase range around burst
        Inputs:
            burst - Data array to be fit
            mode - Desired type of fit, can be gaussian or vonmises
            n - Number of components to be fit
            llimit - Integer, lower limit of phase range to be fit
            hlimit - Integer, upper limit of phase range to be fit
            freq - Frequency of data array, for plotting
            tag - Burst name, e.g. 11A, for plotting
            plot - Boolean, true to make plot and see accuracy of fit
        Returns:
            amp - Array of component amplitudes
            mu - Array of component centers
            retval - Gaussian curve array
    '''
    x = np.linspace(start=llimit, stop=hlimit, num=(hlimit-llimit))
    amp = []
    mu = []
    ForceFit = u.fit_components(xdata = x, ydata = burst[llimit:hlimit], mode = mode,  N=n)    # Forces a fit of n components
    pfit = ForceFit[2]              # Fit parameters
    retval = np.zeros(len(x))
    for j in range(n):
        retval += u.gaussian(x, pfit[3*j], pfit[3*j+1], pfit[3*j+2])        # Add gaussian arrays together for plotting
        amp.append(pfit[3*j])      # Append individual gaussian parameters for analysis 
        mu.append(pfit[3*j+1])
    if plot == True:
        plt.plot(x, retval, 'k')
        plt.plot(x, burst[llimit:hlimit])
        plt.xlabel('Phase Bins')
        plt.ylabel('Flux Density')
        plt.title(tag + ' Peak Flux (at ' + str(round(freq)) + ' MHz)')
        plt.savefig(tag + '_Fit_Test')
    return(amp, mu, retval)

def data_plot(data, fax, tag, center):
    ''' 
        Makes 3D data plot of entire data file, x is phase, y is frequency, z is flux density
        Inputs:
            data - Array of data arrays
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
            center - Array of component centers from comp_param()
        Returns:
            nothing
    '''
    plt.imshow(X = data, aspect = 'auto', interpolation = 'nearest', origin = 'lower', extent = [0,512,fax[0],fax[len(fax)-1]])
    plt.xlabel('Phase Bins')
    plt.ylabel('Frequency(MHz)')
    plt.title('Burst ' + tag + ', Dead Channels Removed')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density')
    if len(center) > 0:
        for i in range(0, len(center)):
            plt.plot(center[i], fax, 'm')
        plt.savefig(tag + '_Data_Center')
    else:
        plt.savefig(tag + '_Data')
    cbar.remove()

def comp_plot(data, name, fax, units, tag, labels, log):
    ''' 
        Makes 2D plot of component parameters vs frequency
        Inputs:
            data - Array of component parameters
            name - Name of parameter to be plotted, e.g. Amplitude
            fax - Frequency axis array
            units - String, y axis units
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
        plt.title(name + ' Versus Frequency of Components of Burst ' + tag)
        plt.savefig(tag + '_Log_' + name)
    else:
        plt.xlabel('Frequency(MHz)')
        plt.ylabel(name + '(' + units + ')')
        plt.title(name + ' Versus Frequency of Components of Burst ' + tag)
        plt.savefig(tag + '_' + name)

def burst_11A_prop():
    '''
        Inputs:
            nothing
        Returns:
            params - Array of component parameters for burst 11A
    '''
    new = start(filename = '11A_16sec.calib.4p')
    tag = '11A'
    PhaseLowLims = [350, 363, 380, 395]
    PhaseHighLims = [362, 370, 390, 420]
    FreqLowLims = [11, 10, 17, 33]
    FreqHighLims = [26, 35, 47, 52]
    params = comp_param(data = new[1], mode = 'gaussian', n = 4, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, fax = new[2], tag = tag)
    return(params)

def main():
    new = start(filename = '12B_743sec.calib.4p')
    tag = '12B'
    PhaseLowLims = [93, 125, 160, 0]
    PhaseHighLims = [113, 150, 0, 0]
    FreqLowLims = [ 11, 28, 0, 0]
    FreqHighLims = [25, 45, 0, 0]
    params = comp_param(data = new[1], mode = 'gaussian', n = 2, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, fax = new[2], tag = tag)
    offset = [[], []]
    for i in params[1][0]:
        real = i-32
        offset[0].append(real)
    for i in params[1][1]:
        real = i-45
        offset[1].append(real)
    '''
    newav = freq_av(data = new[1][10:45])
    plt.plot(newav)
    plt.xlim(50, 200)
    plt.xlabel('Phase Bins')
    plt.ylabel('Flux Density')
    plt.title('Burst 12B Averaged in Frequency')
    plt.savefig('12B_FreqAv')
    '''
    #fit(burst = new[1][40], mode = 'gaussian', n = 1, llimit = 0, hlimit = 200, freq = new[2][40], tag = tag, plot = True)
    labels = ('Comp 1', 'Comp 2')#, 'Comp 3', 'Comp 4')
    #data_plot(data = new[1], fax = new[2], tag = tag, center = offset)
    #comp_plot(data = params[3][0:2], name = 'Fluence', fax = new[2], units = 'Jy ms', tag = tag, labels = labels, log = False)

main()
