import pypulse as p
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
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
    if filename == '12B_743sec.calib.4p':
        ar.bscrunch(nbins = 3000, factor = 6)
    else:
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
        Returns:
            peak - Peak flux value
            burst - Data array containing peak flux value
            freq_index - Array index of input data containing peak value
            tbin - Time bin index of peak flux value
    '''
    peak = 0        # Peak Value
    burst = []      # Burst Data Array
    freq_index = 0        # Frequency Index
    tbin = 0
    for i in range(0,len(data)):          # Move Through Frequency Channels
        for j in range(0,len(data[i])):   # Move Through Phase Bins
            if data[i][j] > peak:
                peak = data[i][j]
                burst = data[i]
                freq_index = i
                tbin = j
    return(peak, burst, freq_index, tbin)

def burst_prop(burst):
    '''
        Finds basic burst properties of input data array (single frequency channel)
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

def freq_av(data, tag, plot, xlims):
    ''' 
        Simple average in frequency
        Inputs:
            data - Array of data arrays
            tag - Burst Name, e.g. 11A
            plot - Boolean, True to make plot
            xlims - Tuple of phase range to zoom in on plot
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
    if plot == True:
        plt.plot(av)
        plt.xlim(xlims[0], xlims[1])
        plt.xlabel('Phase Bins')
        plt.ylabel('Flux Density')
        plt.title('Burst ' + tag + ', Averaged in Frequency')
        plt.savefig(tag + '_FreqAv')
    return(av)

def moments(data):
    '''
        Finds statistical moments (mean, standard deviation, skew, kurtosis) of input distributions or data
        Inputs:
            data - Array of distribution or data arrays
        Returns:
            mus - First moment of distribution, wieghted mean array
            stdevs - Second moment of dsitribution, standard deviation array
            skews - Third moment of distribution, skew array
            kurtoses - Fourth moment of distribution, kurtosis array
    '''
    mus = []
    stdevs = []
    skews = []
    kurtoses = []
    for i in range(0, len(data)):
        mucount = 0
        tot = 0
        N = 0
        for j in range(0, len(data[i])):
            if data[i][j] == 0:
                pass
            else:
                mucount += j*data[i][j]
                tot += data[i][j]
                N += 1
        mu = mucount/tot
        mus.append(mu)
        muind = int(np.round(mu))
        sigcount = 0
        for j in range(0, len(data[i])):
            if data[i][j] == 0:
                pass
            else:
                sigcount += data[i][j]*((j-mu)**2)
        var = (sigcount*N)/(tot*(N-1))
        sigma = var**(0.5)
        stdevs.append(sigma)
        hom = high_order_moments(data = data[i], tot = tot, order = 3, sigma = sigma, mu = mu, rets = [])
        skews.append((-1*hom[0]))
        kurtoses.append(hom[1])
    return(mus, stdevs, skews, kurtoses)

def high_order_moments(data, tot, order, sigma, mu, rets):
    '''
        Helper function for moments(data) function. Recursively finds moments of order 3 and higher
        of data distribution.
        Inputs:
            data - Single distribution array to find moments of
            tot - Sum of data array
            order - Starting order of desired moments (usually 3)
            sigma - Standard deviation of distribution array
            mu - Weighted mean of distribution array
            rets - Empty array
        Returns:
            rets - Input empty array with third and fourth moment appended
    '''
    numer = 0
    for i in range(0, len(data)):
        if data[i] == 0:
            pass
        else:
            numer += data[i]*((i-mu)**order)
    moment = numer/(tot*(sigma**order))
    rets.append(moment)
    order += 1
    if order == 5:
        return(rets)
    else:
        return high_order_moments(data, tot, order, sigma, mu, rets)

def KS_test(vals1, vals2, plot, ind, name):
    '''
        Peforms multiple two-sample Kolmogorov-Smirnov tests with input value arrays and can plot one pair of ECDFs
        Inputs:
            vals1 - Array of first value arrays
            vals2 - Array of second value arrays
            plot - Boolean, True for plotting empirical cumulative distribution function (ECDF) of input value arrays as scatterplots
            ind - Integer index of vals1 and vals2 that should be plotted
            name - String, used for naming saved plot, x axis, and title
        Retruns:
            res - Array of KS values for each pair of value arrays in vals1 and vals2
    '''
    res = []
    for i in range(0, len(vals1)):
        iecdf1 = u.ecdf(values = vals1[i])
        iecdf2 = u.ecdf(values = vals2[i])
        if i == ind:
            if plot == True:
                plt.scatter(x = iecdf1[0], y = iecdf1[1])
                plt.scatter(x = iecdf2[0], y = iecdf2[1])
                plt.xlabel(name)
                plt.ylabel('Cumulative Probability')
                plt.legend(labels = ('Single', 'Multi'))
                plt.title(name + ' ECDF')
                plt.savefig(name[0:2] + 'ecdf')
        diffs = []
        for j in range(0, len(iecdf1[0])):
            if j >= len(iecdf2[0]):
                break
            el = i
            en = i
            while iecdf1[0][j] > iecdf2[0][el]:
                diff = iecdf1[1][j] - iecdf2[1][el]
                diffs.append(-1*diff)
                el += 1
            while iecdf2[0][j] > iecdf1[0][en]:
                diff = iecdf2[1][j] - iecdf1[1][en]
                diffs.append(-1*diff)
                en += 1
                if en >= len(iecdf1[0]):
                    break
        res.append(np.max(diffs))
    return(res)

def SN_reducer(data, peak, SN, desiredSN):
    '''
        Reduces signal to noise of input data, will reduce to about desiredSN value but not exact because
        of random number generation for added noise
        Inputs:
            data - Input data, array of arrays
            peak - Peak flux density, not converted to real units
            SN - Original signal to noise of data set
            desiredSN - Desired signal to noise
        Returns:
            reduced - Original burst data set with increased noise values
    '''
    SNfrac = desiredSN/SN
    reduced = []
    for i in range(0, len(data)):
        newdat = []
        for j in range(0, len(data[i])):
            datdiff = peak - data[i][j]
            ran = np.random.random()
            noise = datdiff*SNfrac*ran
            newdat.append(data[i][j]+(noise/2))
        reduced.append(newdat)
    return(reduced)

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
        Removes all frequencies greater than input - use only for simple analyses of extremely noisy bursts
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
        Removes all frequencies lower than input - use only for simple analyses of extremely noisy bursts
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

def comp_param(data, mode, n, pllim, phlim, fllim, fhlim, factor, fax, tag):
    ''' 
        Finds parameters of burst components
        Inputs:
            data - Array of data arrays
            mode - Desired type of fit, gaussian or vonmises
            n - Number of components to be fit
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
    fluence = [ [], [], [], [] ]
    for i in range(0, len(data)):
        GetFit = fit(burst = data[i], mode = mode, n = n, llimit = pllim[0], hlimit = phlim[n-1], freq = fax[i], tag = tag, plot = [])   # Automatic fit routine
        for j in range(0, len(GetFit[1])):
            if pllim[0] < GetFit[1][j] < phlim[0] and (len(mus[0]) - 1) < i and fllim[0] < i < fhlim[0]:         # Check if component center is within given phase limits and frequency limits
                x = burst_prop(data[i][(pllim[0]-3):pllim[1]])
                widths[0].append(x[1])
                amps[0].append(GetFit[0][j])
                mus[0].append(GetFit[1][j])
                fluence[0].append(np.sum(GetFit[2][0:(phlim[0]-pllim[0])])/(1000*factor))                       # Sum area under fit curve then convert\
            elif pllim[1] < GetFit[1][j] < phlim[1] and (len(mus[1]) - 1) < i and fllim[1] < i < fhlim[1]:      # from flux density and phase bin units to Jy*ms
                x = burst_prop(data[i][(phlim[0]+1):pllim[2]])    # Find FWHM using previous component high limit and next component low limit\
                widths[1].append(x[1])                            # Giving extra noise around component provides more accurate FWHM
                amps[1].append(GetFit[0][j])
                mus[1].append(GetFit[1][j])
                fluence[1].append(np.sum(GetFit[2][(pllim[1]-pllim[0]):(phlim[1]-pllim[0])])/(1000*factor))     # All index ranges of fluence calculation offset by pllim[0]\
            elif pllim[2] < GetFit[1][j] < phlim[2] and (len(mus[2]) - 1) < i and fllim[2] < i < fhlim[2]:      # because that is where the fit starts
                x = burst_prop(data[i][(phlim[1]+1):pllim[3]])
                widths[2].append(x[1])
                amps[2].append(GetFit[0][j])
                mus[2].append(GetFit[1][j])
                fluence[2].append(np.sum(GetFit[2][(pllim[2]-pllim[0]):(phlim[2]-pllim[0])])/(1000*factor))
            elif pllim[3] < GetFit[1][j] < phlim[3] and (len(mus[3]) - 1) < i and fllim[3] < i < fhlim[3]:
                x = burst_prop(data[i][(phlim[2]+1):(phlim[3]+1)])
                widths[3].append(x[1])
                amps[3].append(GetFit[0][j])
                mus[3].append(GetFit[1][j])
                fluence[3].append(np.sum(GetFit[2][(pllim[3]-pllim[0]):(phlim[3]-pllim[0])])/(1000*factor))
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

def lognorm(x, amp, mu, sigma):
    '''
        Creates log-normal distribution with input variables
        Inputs:
            x - x axis
            amp - Amplitude of distribution
            mu - Center of distribution
            sigma - Standard deviation of distribution
        Returns:
            Array of log-normal curve values
    '''
    return amp*np.exp(-0.5*((np.log(x)-mu)/sigma)**2)

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
            plot - Tuple of time conversion (from phase bins) and flux conversion IN THAT ORDER. Used to make time series plot and see accuracy of fit.
                   Enter empty array for no plotting.
        Returns:
            amp - Array of component amplitudes
            mu - Array of component centers
            retval - Gaussian curve array
    '''
    x = np.linspace(start = llimit, stop = hlimit, num = (hlimit-llimit))
    amp = []
    mu = []
    ForceFit = u.fit_components(xdata = x, ydata = burst[llimit:hlimit], mode = mode,  N=n)    # Forces a fit of n components
    pfit = ForceFit[2]              # Fit parameters
    retval = np.zeros(len(x))
    for j in range(n):
        retval += u.gaussian(x, pfit[3*j], pfit[3*j+1], pfit[3*j+2])        # Add gaussian arrays together for plotting
        amp.append(pfit[3*j])      # Append individual gaussian parameters for analysis 
        mu.append(pfit[3*j+1])
    if len(plot) > 0:
        TimeConversion = plot[0]
        FluxConversion = plot[1]
        plt.plot(x/TimeConversion, retval/FluxConversion, 'k')
        plt.plot(x/TimeConversion, burst[llimit:hlimit]/FluxConversion)
        plt.xlabel('Time (ms)')
        plt.ylabel('Flux Density (mJy)')
        plt.title(tag + ' Peak Flux (at ' + str(round(freq)) + ' MHz)')
        plt.savefig(tag + '_Fit_Test')
    return(amp, mu, retval)

def lnorm_fit(xin, burst, n, plot, dattype, units, fax, comp):
    '''
        Fit a Log-Normal curve to a data array and can plot
        Inputs:
            xin - Array of frequency range to be fit
            burst - Data array to be fit
            n - Integer number of curve fits desired, fit is better with greater n
            plot - Boolean, True to make plot with data and fit
            dattype - Type of data being plotted, string used for y axis label and title
            units - Unit of data type, string used for y axis label
            fax - Frequency axis of data array
            comp - Component label, for title and saving. Must be string of form: Component #
        Returns:
            res[0] - Resulting amplitude, mean, and standard dev of fit, to be used in lognormal plot
    '''
    bins = len(xin)
    maxind = np.argmax(burst)
    init = np.array([burst[maxind], np.log(xin[maxind]), 2])        # Initial Guess for fit uses the max amplitude indices and a sigma of 2
    for i in range(1,n+1):
        def fitter(x, p):               # Fit Function for lognorm, this goes into leastsq
            pdf = np.zeros(len(x))
            for j in range(i):
                pdf += lognorm(x, p[3*j], p[3*j+1], p[3*j+2])
            return(pdf)
        def err(p, x, y):               # Error Function, this is used in leastsq to iteratively fit a number of components until the smallest sum of squares is found
            return y - fitter(x, p)
        res = sci.optimize.leastsq(func = err, x0 = init, args = (xin, burst))      # Finds The smallest sum of squares for fit and data
        if i == n:
            break
        pfit = res[0]
        result = burst - fitter(xin, pfit)
        newinit = np.array([result[maxind], np.log(xin[maxind]), 2])            # Redefine initial conditions for next component
        init = np.concatenate((pfit, newinit))
    if plot == True:
        pdf = np.zeros(len(xin))
        for i in range(n):
         pdf += lognorm(x = xin, amp = res[0][3*i], mu = res[0][3*i+1], sigma = res[0][3*i+2])  # Add up all components to get total fit
        plt.plot(fax[1:], burst)
        plt.plot(fax[1:], pdf)
        plt.legend(labels = (dattype, 'Log-Normal Fit'))
        plt.xlabel('Frequency(MHz)')
        plt.ylabel(dattype + units)
        plt.title('Log-Normal fit to ' + dattype + ' of ' + comp)
        plt.savefig(comp[0:4] + comp[-1] + '_LNorm_Fit')
    return(res[0])

def gauss_lnorm_fit(xin, burst, dattype, units, fax, comp):
    '''
        Fit a single gaussian plus a log-normal curve to the data array and makes plot, for very sharply peaked data
        Inputs:
            xin - Array of frequency range to be fit
            burst - Data array to be fit
            dattype - Type of data being plotted, string used for y axis label and title
            units - Unit of data type, string used for y axis label
            fax - Frequency axis of data array
            comp - Component label, for title and saving. Must be string of form: Component #
        Returns:
            nothing
    '''
    gauss = fit(burst = burst, mode = 'gaussian', n = 1, llimit = 0, hlimit = 64, freq = 0, tag = '', plot = False)     # Get gaussian fit of data
    resid = burst - gauss[2]            # Find residual of data minus gaussian fit
    pfit = lnorm_fit(xin = xin, burst = resid[1:], n = 1, plot = False, dattype = dattype, units = units, fax = fax, comp = comp)   # Get log-normal fit parameters of residual
    pdf = np.zeros(len(xin))
    pdf += lognorm(x = xin, amp = pfit[0], mu = pfit[1], sigma = pfit[2])   # Make log-normal curve
    FullFit = gauss[2][1:] + pdf        # Add gaussian and log-normal curve to get combined fit curve
    plt.plot(fax[1:], burst[1:])
    plt.plot(fax[1:], FullFit)
    plt.legend(labels = (comp, 'Gauss+LogNormal Fit'))
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(dattype + units)
    plt.title('Gauss Plus LogNormal fit to ' + dattype + ' of ' + comp)
    plt.savefig(comp[0:4] + comp[-1] + '_GaLNorm_Fit')

def data_plot(data, fax, tag, center, RSN, vmax, ext=512):
    ''' 
        Makes 3D data plot of entire data file, x is phase, y is frequency, z is flux density
        Inputs:
            data - Array of data arrays
            fax - Frequency axis array
            tag - Burst name, e.g. 11A
            center - Array of component centers from comp_param(), input empty array if no center
            RSN - Boolean, True if S/N of burst is reduced
            vmax - Maximum value of data plot, 0 if automatic interpolation desired
            ext - Length of horizontal axis, automatically set to 512 unless specified
        Returns:
            nothing
    '''
    if vmax > 0:
        plt.imshow(X = data, aspect = 'auto', interpolation = 'nearest', origin = 'upper', vmin = 0, vmax = vmax, extent = [0, ext, fax[len(fax)-1], fax[0]])
    else:
        plt.imshow(X = data, aspect = 'auto', interpolation = 'nearest', origin = 'upper', extent = [0,ext,fax[len(fax)-1],fax[0]])
    plt.xlabel('Time(ms)')
    plt.ylabel('Frequency(MHz)')
    if RSN == True:
        plt.title('Burst ' + tag[0:3] + ', SN Reduced')
    else:
        plt.title('Burst ' + tag + ', Dead Channels Removed')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density (mJy)')
    if len(center) > 0:
        for i in range(0, len(center)):
            plt.plot(center[i], fax, 'm')
        if RSN == True:
            plt.savefig(tag + '_Reduced_Data_Center')
        else:
            plt.savefig(tag + '_Data_Center')
    else:
        if RSN == True:
            plt.savefig(tag + '_Reduced_Data')
        else:
            plt.savefig(tag + '_Data')
    cbar.remove()

def comp_plot(data, name, fax, units, tag, labels, log, RSN):
    ''' 
        Makes 2D plot of a component parameter vs frequency
        Inputs:
            data - Array of component parameters
            name - Name of parameter to be plotted, e.g. Amplitude
            fax - Frequency axis array
            units - String, y axis units
            tag - Burst name, e.g. 11A
            labels - Component name list for legend
            log - Boolean, True for log plot
            RSN - Boolean, True if S/N of burst is reduced
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
        if RSN == True:
            plt.title(name + ' vs. Frequency of Components of Reduced S/N Burst ' + tag[0:3])
            plt.savefig(tag + '_Reduced_' + name)
        else:
            plt.title(name + ' Versus Frequency of Components of Burst ' + tag)
            plt.savefig(tag + '_' + name)

def moment_hist(vals, xname, pname, multicomp):
    '''
        Creates a histogram of input moment values
        Inputs:
            vals - Moment value array
            xname - Moment name string, used for x axis label, title, and saving
            pname - Parameter name string, used for saving
            multicomp - Boolean, true for making plot of multiple components
        Returns:
            nothing
    '''
    plt.hist(vals, bins = 10)
    plt.xlabel(xname)
    if multicomp == True:
        plt.title(xname + ' of Component ' + pname)
        plt.savefig('Multi_' + pname + '_' + xname[0:4])
    else:
        plt.title(xname + ' of Burst ' + pname)
        plt.savefig('Single_' + pname + '_' + xname[0:4])
    plt.clf()

def fluence_moment_scatt(tfdmarr, moment, RSN, singleA):
    '''
        Creates a scatter plot of total fluence versus fluence moments (standard deviation, skew, and kurtosis)
        Inputs:
            tfdmarr - Burst information, array of arrays containing given tag, fluence array, data array, and moment array
            moment - Desired moment to plot on x axis
            RSN - Boolean, True if S/N of bursts 11A and 12B are reduced
            singleA - Boolean, True if burst 11A is fit with a single component
        Returns:
            Nothing
    '''
    moments = []
    fluences = []
    Extmoments = []
    Extfluences = []
    for i in range(0, len(tfdmarr)):
        if tfdmarr[i][0] == '11A':
            Amoments = []
            Afluences = []
            for j in range(0, len(tfdmarr[i][1])):
                Afluences.append(np.sum(tfdmarr[i][1][j]))
                if moment == 'SD':
                    Amoments.append(tfdmarr[i][3][1][j])
                elif moment == 'Skew':
                    Amoments.append(tfdmarr[i][3][2][j])
                elif moment == 'Kurtosis':
                    Amoments.append(tfdmarr[i][3][3][j])
        elif tfdmarr[i][0] == '12B':
            Bmoments = []
            Bfluences = []
            for j in range(0, len(tfdmarr[i][1])):
                Bfluences.append(np.sum(tfdmarr[i][1][j]))
                if moment == 'SD':
                    Bmoments.append(tfdmarr[i][3][1][j])
                elif moment == 'Skew':
                    Bmoments.append(tfdmarr[i][3][2][j])
                elif moment == 'Kurtosis':
                    Bmoments.append(tfdmarr[i][3][3][j])
        elif len(tfdmarr[i][0]) > 3:
            Extfluences.append(np.sum(tfdmarr[i][1]))
            if moment == 'SD':
                Extmoments.append(tfdmarr[i][3][1])
            elif moment == 'Skew':
                Extmoments.append(tfdmarr[i][3][2])
            elif moment == 'Kurtosis':
                Extmoments.append(tfdmarr[i][3][3])
        else:
            fluences.append(np.sum(tfdmarr[i][1]))
            if moment == 'SD':
                moments.append(tfdmarr[i][3][1])
            elif moment == 'Skew':
                moments.append(tfdmarr[i][3][2])
            elif moment == 'Kurtosis':
                moments.append(tfdmarr[i][3][3])
    plt.scatter(x = Amoments, y = Afluences, c = 'orange', marker = '*')
    plt.scatter(x = Bmoments, y = Bfluences, c = 'black', marker = 'D')
    plt.scatter(x = Extmoments, y = Extfluences, c = 'red', marker = 'h')
    plt.scatter(x = moments, y = fluences)
    plt.xlabel(moment)
    plt.ylabel('Total Fluence(Jy ms)')
    if RSN == True:
        plt.title('Total Fluence vs. ' + moment + ' with S/N of Bursts 11A and 12B Reduced')
        if singleA == True:
            plt.legend(labels = ('Burst 11A, Single Comp', 'Burst 12B', 'Low S/N Bursts', 'Single Comp Bursts'))
            plt.savefig('Reduced1_FvM' + moment)
        else:
            plt.legend(labels = ('Burst 11A', 'Burst 12B', 'Low S/N Bursts', 'Single Comp Bursts'))
            plt.savefig('Reduced_FvM' + moment)
    else:
        plt.title('Total Fluence vs. ' + moment)
        plt.legend(labels = ('Burst 11A', 'Burst 12B', 'Low S/N Bursts', 'Single Comp Bursts'))
        plt.savefig('FvM' + moment)

def burst_11A_prop():
    '''
        Uses manually determined frequency and phase ranges to output burst 11A component parameters
        Inputs:
            nothing
        Returns:
            params - Array of component parameters for burst 11A
            data - Full data of burst after noise reduction, array of frequency data arrays
            smax - Maximum signal of burst in mJy
            fax - Frequency axis array
    '''
    new = start(filename = '11A_16sec.calib.4p')
    data = new[1]
    tag = '11A'
    smax = 380.5
    fax = new[2]
    PhaseLowLims = [350, 363, 380, 395]
    PhaseHighLims = [362, 370, 390, 420]
    FreqLowLims = [11, 10, 17, 33]
    FreqHighLims = [26, 35, 47, 52]
    peak = find_peak(data[FreqLowLims[0]:FreqHighLims[3]])
    factor = (peak[0]/smax)*40              # Conversion of peak flux density to mJy by dividing by the max signal in mJy then multiplied by phase bin conversion to ms
    params = comp_param(data = data, mode = 'gaussian', n = 4, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, factor = factor, fax = fax, tag = tag)
    return(params, data, smax, fax)

def burst_12B_prop():
    '''
        Uses manually determined frequency and phase ranges to output burst 12B component parameters
        Inputs:
            nothing
        Returns:
            params - Array of component parameters for burst 12B
            data - Full data of burst after noise reduction, array of frequency data arrays
            smax - Maximum signal of burst in mJy
            fax - Frequncy axis array
    '''
    new = start(filename = '12B_743sec.calib.4p')
    data = new[1]
    tag = '12B'
    smax = 331.4
    fax = new[2]
    PhaseLowLims = [65, 85, 115, 0]            # Must include a third lower limit for width calculation
    PhaseHighLims = [75, 105, 0, 0]
    FreqLowLims = [ 11, 28, 0, 0]
    FreqHighLims = [25, 45, 0, 0]
    peak = find_peak(data[FreqLowLims[0]:FreqHighLims[1]])
    factor = (peak[0]/smax)*27          # Smaller phase bin conversion because this burst has a larger bscrunch
    params = comp_param(data = data, mode = 'gaussian', n = 2, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, factor = factor, fax = fax, tag = tag)
    return(params, data, smax, fax)

def unres_comp_prop(tag, single):
    '''
        Returns burst parameters of possibly unresolved component bursts as either single component
        or as the two most likely components using manually determined frequency and phase ranges
        Inputs:
            tag - Burst name of desired burst parameters, e.g. 11E
            single - Boolean, if true retrieve burst parameters as a single component
        Returns:
            params - Array of component parameters for desired burst
            data - Full data of burst after noise redusction, array of frequency arrays
            smax - Maximum signal of burst in mJy
            fax - Frequency axis array
    '''
    if tag == '11E':
        new = start(filename = '11E_344sec.calib.4p')
        smax = 126.8
        if single == False:
            PhaseLowLims = [360, 383, 420, 0]
            PhaseHighLims = [382, 415, 0, 0]
            FreqLowLims = [12, 28, 0, 0]
            FreqHighLims = [23, 49, 0, 0]
        elif single == True:
            PhaseLowLims = [360, 420, 0, 0]
            PhaseHighLims = [415, 0, 0, 0]
            FreqLowLims = [12, 0, 0, 0]
            FreqHighLims = [49, 0, 0, 0]
    elif tag == '11K':
        new = start(filename = '11K_769sec.calib.4p')
        smax = 105.2
        if single == False:
            PhaseLowLims = [393, 402, 427, 0]
            PhaseHighLims = [401, 422, 0, 0]
            FreqLowLims = [0, 7, 0, 0]
            FreqHighLims = [4, 22, 0, 0]
        elif single == True:
            PhaseLowLims = [393, 427, 0, 0]
            PhaseHighLims = [422, 0, 0, 0]
            FreqLowLims = [0, 0, 0, 0]
            FreqHighLims = [22, 0, 0, 0]
    elif tag == '11O':
        new = start(filename = '11O_1142sec.calib.4p')
        smax = 139.9
        if single == False:
            PhaseLowLims = [460, 469, 495, 0]
            PhaseHighLims = [468, 490, 0, 0]
            FreqLowLims = [24, 38, 0, 0]
            FreqHighLims = [37, 46, 0, 0]
        elif single == True:
            PhaseLowLims = [460, 495, 0, 0]
            PhaseHighLims = [490, 0, 0, 0]
            FreqLowLims = [24, 0, 0, 0]
            FreqHighLims = [46, 0, 0, 0]
    else:
        raise NameError('Tag Not Found')
    data = new[1]
    if single == False:
        peak = find_peak(data[FreqLowLims[0]:FreqHighLims[1]])
        n = 2
    elif single == True:
        peak = find_peak(data[FreqLowLims[0]:FreqHighLims[0]])
        n = 1
    factor = (peak[0]/smax)*40
    fax = new[2]
    params = comp_param(data = data, mode = 'gaussian', n = n, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, factor = factor, fax = fax, tag = tag)
    return(params, data, smax, fax)

def single_comp_prop(tag):
    '''
        Returns burst parameters of single component bursts using the manually determined frequency and phase ranges
        Inputs:
            tag - Burst name of desired burst parameters, e.g. 11B
        Returns:
            params - Array of component parameters for desired single component burst
            data - Full data of burst after noise reduction, array of frequency data arrays
            smax - Maximum signal of burst in mJy
            fax - Frequency axis array
    '''
    if tag == '11B':
        new = start(filename = '11B_263sec.calib.4p')
        PhaseLowLims = [72, 90, 0, 0]       # All must include a second lower limit for width calculation
        PhaseHighLims = [84, 0, 0, 0]
        FreqLowLims = [37, 0, 0, 0]
        FreqHighLims = [48, 0, 0, 0]
        smax = 51.9
    elif tag == '11C':
        new = start(filename = '11C_284sec.calib.4p')
        PhaseLowLims = [125, 145, 0, 0]
        PhaseHighLims = [140, 0, 0, 0]
        FreqLowLims = [10, 0, 0, 0]
        FreqHighLims = [21, 0, 0, 0]
        smax = 85.2
    elif tag == '11D':
        new = start(filename = '11D_323sec.calib.4p')
        PhaseLowLims = [5, 40, 0, 0]
        PhaseHighLims = [35, 0, 0, 0]
        FreqLowLims = [11, 0, 0, 0]
        FreqHighLims = [35, 0, 0, 0]
        smax = 314.8
    elif tag == '11F':
        new = start(filename = '11F_356sec.calib.4p')
        PhaseLowLims = [455, 500, 0, 0]
        PhaseHighLims = [490, 0, 0, 0]
        FreqLowLims = [10, 0, 0, 0]
        FreqHighLims = [22, 0, 0, 0]
        smax = 157.5
    elif tag == '11G':
        new = start(filename = '11G_580sec.calib.4p')
        PhaseLowLims = [290, 310, 0, 0]
        PhaseHighLims = [305, 0, 0, 0]
        FreqLowLims = [40, 0, 0, 0]
        FreqHighLims = [48, 0, 0, 0]
        smax = 52.9
    elif tag == '11H':
        new = start(filename = '11H_597sec.calib.4p')
        PhaseLowLims = [12, 37, 0, 0]
        PhaseHighLims = [32, 0, 0, 0]
        FreqLowLims = [0, 0, 0, 0]
        FreqHighLims = [23, 0, 0, 0]
        smax = 699.9
    elif tag == '11I':
        new = start(filename = '11I_691sec.calib.4p')
        PhaseLowLims = [138, 165, 0, 0]
        PhaseHighLims = [160, 0, 0, 0]
        FreqLowLims = [32, 0, 0, 0]
        FreqHighLims = [56, 0, 0, 0]
        smax = 125.5
    elif tag == '11J':
        new = start(filename = '11J_704sec.calib.4p')
        PhaseLowLims = [460, 485, 0, 0]
        PhaseHighLims = [480, 0, 0, 0]
        FreqLowLims = [17, 0, 0, 0]
        FreqHighLims = [40, 0, 0, 0]
        smax = 118.6
    elif tag == '11M':
        new = start(filename = '11M_993sec.calib.4p')
        PhaseLowLims = [253, 267, 0, 0]
        PhaseHighLims = [264, 0, 0, 0]
        FreqLowLims = [11, 0, 0, 0]
        FreqHighLims = [21, 0, 0, 0]
        smax = 94.7
    elif tag == '11N':
        new = start(filename = '11N_1036sec.calib.4p')
        PhaseLowLims = [45, 65, 0, 0]
        PhaseHighLims = [60, 0, 0, 0]
        FreqLowLims = [41, 0, 0, 0]
        FreqHighLims = [60, 0, 0, 0]
        smax = 256.7
    elif tag == '11Q':
        new = start(filename = '11Q_1454sec.calib.4p')
        PhaseLowLims = [57, 73, 0, 0]
        PhaseHighLims = [68, 0, 0, 0]
        FreqLowLims = [29, 0, 0, 0]
        FreqHighLims = [47, 0, 0, 0]
        smax = 400.2
    elif tag == '12A':
        new = start(filename = '12A_104sec.calib.4p')
        PhaseLowLims = [427, 470, 0, 0]
        PhaseHighLims = [457, 0, 0, 0]
        FreqLowLims = [33, 0, 0, 0]
        FreqHighLims = [48, 0, 0, 0]
        smax = 132.2
    elif tag == '12C':
        new = start(filename = '12C_1526sec.calib.4p')
        PhaseLowLims = [305, 350, 0, 0]
        PhaseHighLims = [340, 0, 0, 0]
        FreqLowLims = [36, 0, 0, 0]
        FreqHighLims = [52, 0, 0, 0]
        smax = 118.8
    else:
        raise NameError('Tag Not Found')
    data = new[1]
    peak = find_peak(data[FreqLowLims[0]:FreqHighLims[0]])
    factor = (peak[0]/smax)*40          # All single component bursts have the same phase bin conversion as burst 11A
    fax = new[2]
    params = comp_param(data = data, mode = 'gaussian', n = 1, pllim = PhaseLowLims, phlim = PhaseHighLims, fllim = FreqLowLims, fhlim = FreqHighLims, factor = factor, fax = fax, tag = tag)
    return(params, data, smax, fax)

def burst_stats(multi, plot):
    '''
        Can find statistical moments of each burst in breakthrough listen original 21 burst data set
        Inputs:
            multi - Boolean for desired burst type, True to get moments of each component of multiple component bursts
                    and perform the same analysis for unresolved components as if they were multiple separate components
                    False to get moments of single component bursts
            plot - Boolean for plotting, True to make histogram of moments
        Returns:
            stdev - Standard deviations of desired burst type, array
            skews - Skews of desired burst type, array
            kurt - Kurtoses of desired burst type, array
            tfdmarr - Array of burst info. Each element is an array of: [tag, fluence array, data array, moment array]
    '''
    tags = ['11B', '11C', '11D', '11E', '11F', '11G', '11H', '11I', '11J', '11K', '11M', '11N', '11O', '11Q', '12A', '12C']
    multitags = ['11A', '12B', '11E', '11K', '11O']
    stdev = []
    skews = []
    kurt = []
    tfdmarr = []
    if multi == True:
        arr = multitags
        single = False
        cutoff = 2
    else:
        arr = tags
        single = True
        cutoff = 1
    for j in range(0, len(arr)):
        if arr[j] == '11A':
            props = burst_11A_prop()
            fluence = props[0][3]
            moms = moments(data = fluence)
            for k in range(0, len(moms[0])):
                stdev.append(moms[1][k])
                skews.append(moms[2][k])
                kurt.append(moms[3][k])
        elif arr[j] == '12B':
            props = burst_12B_prop()
            fluence = props[0][3][0:2]
            moms = moments(data = fluence)
            for k in range(2):
                stdev.append(moms[1][k])
                skews.append(moms[2][k])
                kurt.append(moms[3][k])
        elif arr[j] == '11E':
            props = unres_comp_prop(tag = '11E', single = single)
            fluence = props[0][3][0:cutoff]
            moms = moments(data = fluence)
            for k in range(cutoff):
                stdev.append(moms[1][k])
                skews.append(moms[2][k])
                kurt.append(moms[3][k])
        elif arr[j] == '11K':
            props = unres_comp_prop(tag = '11K', single = single)
            fluence = props[0][3][0:cutoff]
            moms = moments(data = fluence)
            for k in range(cutoff):
                stdev.append(moms[1][k])
                skews.append(moms[2][k])
                kurt.append(moms[3][k])
        elif arr[j] == '11O':
            props = unres_comp_prop(tag = '11O', single = single)
            fluence = props[0][3][0:cutoff]
            moms = moments(data = fluence)
            for k in range(cutoff):
                stdev.append(moms[1][0])
                skews.append(moms[2][0])
                kurt.append(moms[3][0])
        else:
            props = single_comp_prop(tag = arr[j])
            fluence = [props[0][3][0]]
            moms = moments(data = fluence)
            stdev.append(moms[1][0])
            skews.append(moms[2][0])
            kurt.append(moms[3][0])
        tfdm = [arr[j], fluence, props[1], moms]
        tfdmarr.append(tfdm)
    if plot == True:
        ParamName = 'Fluence'
        moment_hist(vals = stdev, xname = 'Standard Deviation', pname = ParamName, multicomp = multi)
        moment_hist(vals = skews, xname = 'Skew', pname = ParamName, multicomp = multi)
        moment_hist(vals = kurt, xname = 'Kurtosis', pname = ParamName, multicomp = multi)
    else:
        return(stdev, skews, kurt, tfdmarr)

def SN_homogenize(reducee, plot):
    '''
        Reduces signal to noise of bursts 11A and 12B to about the average of the other 19 bursts in BL original data set
        The reduced S/N is not exactly the average because of random number generation for added noise
        Inputs:
            reducee - The tag of the burst having S/N reduced, e.g. '11A'
            plot - Boolean, True to make plot of reduced S/N data
        Returns:
            reduced - Reduced S/N data of desired burst
    '''
    if reducee == '11A':
        rinit = burst_11A_prop()
        rpeak = find_peak(data = rinit[1])
        rpeakind = rpeak[2]
        rprops = burst_prop(burst = rinit[1][rpeakind])
        rSN = rprops[2]
        print('Original S/N: ' + str(rSN))
    elif reducee == '12B':
        rinit = burst_12B_prop()
        rpeak = find_peak(data = rinit[1])
        rpeakind = rpeak[2]
        rprops = burst_prop(burst = rinit[1][rpeakind])
        rSN = rprops[2]
        print('Original S/N: ' + str(rSN))
    else:
        raise NameError('Tag Not Found')
    tags1 = ['11E', '11K', '11O']
    count = 0
    for i in range(0, len(tags1)):
        init = unres_comp_prop(tag = tags1[i], single = True)
        peakinit = find_peak(data = init[1])
        peakind = peakinit[2]
        props = burst_prop(burst = init[1][peakind])
        count += props[2]
    tags2 = ['11B', '11C', '11D', '11F', '11G', '11H', '11I', '11J', '11M', '11N', '11Q', '12A', '12C']
    for i in range(0, len(tags2)):
        init = single_comp_prop(tag = tags2[i])
        peakinit = find_peak(data = init[1])
        peakind = peakinit[2]
        props = burst_prop(burst = init[1][peakind])
        count += props[2]
    desired = count/(len(tags1)+len(tags2))
    print('Desired S/N: ' + str(desired))
    reduced = SN_reducer(data = rinit[1], peak = rpeak[0], SN = rSN, desiredSN = desired)
    newpeak = find_peak(data = reduced)
    newprops = burst_prop(burst = reduced[newpeak[2]])
    print('New S/N: ' + str(newprops[2]))
    if plot == True:
        data_plot(data = reduced, fax = rinit[3], tag = reducee, center = [])
    return(np.array(reduced))

def reduced_SN_props(singleA):
    '''
        Reduces the S/N ratio of bursts 11A and 12B then finds fluence and moments of the fluence distribution
        Inputs:
            SingleA - Boolean, True for burst 11A to be fit as a single component burst and false for it to be
                      fit as a 3 component burst. Burst 12B is always fit with a single component for reduced S/N
        Returns:
            nothing
    '''
    Ainit = burst_11A_prop()
    Apeak = find_peak(data = Ainit[1])
    Apeakind = Apeak[2]
    Aprops = burst_prop(burst = Ainit[1][Apeakind])
    ASN = Aprops[2]
    reducedA = SN_reducer(data = Ainit[1], peak = Apeak[0], SN = ASN, desiredSN = 13.3)
    if singleA == True:
        reducedAprops = comp_param(data = np.array(reducedA), mode = 'gaussian', n = 1, pllim = [350, 425, 0, 0], phlim = [420, 0, 0, 0], fllim = [10, 0, 0, 0], \
                fhlim = [55, 0, 0, 0], factor = (Apeak[0]/Ainit[2])*40, fax = Ainit[3], tag = '11A')
        tfdmA = ['11A', [reducedAprops[3][0]], reducedA, moments(data = [reducedAprops[3][0]])]
    else:
        reducedAprops = comp_param(data = np.array(reducedA), mode = 'gaussian', n = 3, pllim = [350, 380, 395, 425], phlim = [370, 390, 420, 0], fllim = [10, 27, 33, 0], \
                fhlim = [30, 45, 48, 0], factor = (Apeak[0]/Ainit[2])*40, fax = Ainit[3], tag = '11A')
        tfdmA = ['11A', reducedAprops[3][0:3], reducedA, moments(data = reducedAprops[3][0:3])]
    Binit = burst_12B_prop()
    Bpeak = find_peak(data = Binit[1])
    Bpeakind = Bpeak[2]
    Bprops = burst_prop(burst = Binit[1][Bpeakind])
    BSN = Bprops[2]
    reducedB = SN_reducer(data = Binit[1], peak = Bpeak[0], SN = BSN, desiredSN = 13.3)
    reducedBprops = comp_param(data = np.array(reducedB), mode = 'gaussian', n =1, pllim = [65, 115, 0, 0], phlim = [105, 0, 0, 0], fllim = [11, 0, 0, 0], fhlim = [42, 0, 0, 0], \
            factor = (Bpeak[0]/Binit[2])*40, fax = Binit[3], tag = '12B')
    tfdmB = ['12B', [reducedBprops[3][0]], reducedB, moments(data = [reducedBprops[3][0]])]
    fvm = burst_stats(multi = False, plot = False)
    fvm[3].append(tfdmA)
    fvm[3].append(tfdmB)
    fluence_moment_scatt(tfdmarr = fvm[3], moment = 'SD', RSN = True, singleA = singleA)
    plt.clf()
    fluence_moment_scatt(tfdmarr = fvm[3], moment = 'Skew', RSN = True, singleA = singleA)
    plt.clf()
    fluence_moment_scatt(tfdmarr = fvm[3], moment = 'Kurtosis', RSN = True, singleA = singleA)
    plt.clf()

def make_dynamic_spectra(center):
    '''
        Outputs dyanmic spectra of all bursts in Brreakthrough Listen data set
        Inputs:
            center - Boolean, True to overplot the Gaussian fit centers ont components of burst 11A and 12B
        Returns:
            nothing
   '''
    A = burst_11A_prop()
    peak = find_peak(A[1])
    TimeConversion = 25.6   #Phase Bins per millisecond
    FluxConversion = peak[0]/A[2]  #Flux units per milliJansky
    ConvertedData = A[1]/FluxConversion
    if center == True:
        CenterConversion = []
        for i in range(0, len(A[0][1])):
            centarr = []
            for j in range(0, len(A[0][1][i])):
                centarr.append(A[0][1][i][j]/TimeConversion)
            CenterConversion.append(centarr)
        data_plot(data = ConvertedData, fax = A[3], tag = '11A', center = CenterConversion, RSN = False, vmax = 0, ext = 512/TimeConversion)
    else:
        data_plot(data = ConvertedData, fax = A[3], tag = '11A', center = [], RSN = False, vmax = 0, ext = 512/Timeconversion)
    B = burst_12B_prop()
    peak = find_peak(B[1])
    TimeConversion12B = 25  #Burst 12B has a longer horizontal axis
    FluxConversion = peak[0]/B[2]
    ConvertedData = B[1]/FluxConversion
    if center == True:
        CenterConversion12B = []
        for i in range(0, len(B[0][1])):
            centarr = []
            for j in range(0, len(B[0][1][i])):
                centarr.append(B[0][1][i][j]/TimeConversion12B)
            CenterConversion12B.append(centarr)
        data_plot(data = ConvertedData, fax = B[3], tag = '12B', center = CenterConversion12B, RSN = False, vmax = 0, ext = 500/TimeConversion12B)
    else:
        data_plot(data = ConvertedData, fax = B[3], tag = '12B', center = [], RSN = False, vmax = 0, ext = 500/TimeConversion)
    tags = ['11B', '11C', '11D', '11F', '11G', '11H', '11I', '11J', '11M', '11N', '11Q', '12A', '12C']
    for tag in tags:
        burst = single_comp_prop(tag)
        peak = find_peak(burst[1])
        FluxConversion = peak[0]/burst[2]
        ConvertedData = burst[1]/FluxConversion
        data_plot(data = ConvertedData, fax = burst[3], tag = tag, center = [], RSN = False, vmax = 0, ext = 512/TimeConversion)
    unrestags = ['11E', '11K', '11O']
    for utag in unrestags:
        burst = unres_comp_prop(tag = utag, single = True)
        peak = find_peak(burst[1])
        FluxConversion = peak[0]/burst[2]
        ConvertedData = burst[1]/FluxConversion
        data_plot(data = ConvertedData, fax = burst[3], tag = utag, center = [], RSN = False, vmax = 0, ext = 512/TimeConversion)

def main():
    print('Initializing BL21 Burst Code')
    #make_dynamic_spectra(center = True)
    '''
    A = burst_11A_prop()
    peakA = find_peak(A[1])
    TimeConversion = 25.6 #512 phase bins divided by 20 milliseconds
    FluxConversion = peakA[0]/A[2]
    fit(burst = A[1][peakA[2]], mode = 'gaussian', n = 4, llimit = 340, hlimit = 400, freq = A[3][peakA[2]], tag = '11A', plot = [TimeConversion, FluxConversion])
    '''
    #reduced_SN_props(singleA = True)
    #data_plot(data = reducedA[0], fax = reducedA[3], tag = '11A-1', center = reducedAprops1[1], RSN = True)
    #comp_plot(data = [reducedBprops[3][0]], name = 'Fluence', fax = reducedB[3], units = 'Jy ms', tag = '12B, labels = ('Comp1'), log = False, RSN = True)
    '''
    stats1 = burst_stats(multi = False, plot = False)
    stats2 = burst_stats(multi = True, plot = False)
    ks = KS_test(vals1 = stats1, vals2 = stats2, plot = False, ind = 1, name = 'Skew')
    print(ks)
    '''

main()
