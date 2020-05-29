import pypulse as p
import matplotlib.pyplot as plt
import numpy as np
import pypulse.utils as u
import pypulse.rfimitigator as rfim

def start(filename):
    ar = p.Archive(filename)
    ar.dedisperse(reverse = True)
    rm = mitigate(ar)
    #nulow = [8000]
    #nuhigh = [8400]
    #zap_freq(rm, nulows = nulow, nuhighs = nuhigh)
    ar.bscrunch(nbins = 2048, factor = 4)       # Average Burst in Phase
    ar.fscrunch(nchan = 19456, factor = 304)    # Average Burst in Frequency
    data = ar.getData()
    fax = ar.getAxis(flag = 'F')        # Frequency Axis
    return(ar, data, fax)

def find_peak(data):
    ''' Finds Peak Flux Value and Data Array at the Peak Flux Frequency '''
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
    sp = p.SinglePulse(data = burst, windowsize = 256)
    FWHM = sp.getFWHM()
    SN = sp.getSN()
    return(sp, FWHM, SN)

def mitigate(ar):
    ''' Finds and Removes Dead Channels '''
    rm = rfim.RFIMitigator(ar)
    rm.zap_minmax()             #Auto-Zap Dead Channels
    return(rm)

def zap_freq(rm, nulows, nuhighs):      #Array of Ranges Must be Ordered
    ''' Removes Dead Channels'''
    if len(nulows) != len(nuhighs):     #Check for Valid Ranges
        return()
    for i in range(0, len(nulows)):
        rm.zap_frequency_range(nulow = nulows[i], nuhigh = nuhighs[i])
    return(rm)

def unweight(ar, frequencies):
    ''' Sets Statistical Weight to Zero '''
    ar.setWeights(val = 0.0, f = frequencies)
    return(ar)

def destroy_greater(data, index):
    ''' Removes All Frequencies Greater Than Input '''
    for i in range(index, len(data)):
        for j in range(0, len(data[i])):
            data[i][j] = 0.0
    return(data)

def destroy_lower(data, index):
    ''' Removes All Frequencies Lower Than Input '''
    for i in range(0, index):
        for j in range(0, len(data[i])):
            data[i][j] = 0.0
    return(data)

def fit(burst, mode, freq, tag, plot):
    ''' Fits n Components to Data and Can Plot '''
    x = np.linspace(start=1, stop=512, num=512)
    amp = []
    mu = []
    ForceFit = u.fit_components(xdata = x, ydata = burst, mode = mode,  N=4)
    pfit = ForceFit[2]
    retval = np.zeros(len(x))
    for j in range(4):
        retval += u.gaussian(x, pfit[3*j], pfit[3*j+1], pfit[3*j+2])
        amp.append(pfit[3*j])
        mu.append(pfit[3*j+1])
    if plot == True:
        ''' Use for Plotting Peak Flux Frequency '''
        plt.plot(x, retval, 'k')
        plt.plot(x, burst)
        #plt.xlim(340,430)          #Zoom in on data
        plt.xlabel('Phase Bins')
        plt.ylabel('Flux Density')
        plt.title(tag + ' Peak Flux (at ' + str(round(freq)) + ' MHz)')
        plt.savefig(tag + '_Fit')
    return(amp, mu)

def data_plot(data, fax, tag):
    ''' Makes Data Plot of File '''
    plt.imshow(X = data, aspect = 'auto', interpolation = 'nearest', origin = 'lower', extent = [0,512,fax[0],fax[len(fax)-1]])
    plt.xlabel('Phase Bins')
    plt.ylabel('Frequency(MHz)')
    plt.title('Burst ' + tag + ', Dead Channels Removed')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density')
    plt.savefig(tag + '_Data')
    cbar.remove()

def pplot(param, pname, fax, tag):
    ''' Plots Input Fit Parameter vs Frequency '''
    for i in param:
        plt.plot(fax, i)
    labels = ('Comp 1', 'Comp 2', 'Comp 3', 'Comp 4')
    plt.legend(labels = labels)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('Frequency')
    plt.ylabel(pname)
    plt.title(pname + ' Versus Frequency of Components of Burst ' + tag)
    plt.savefig(tag + '_CompMu')

def main():
    ''' Makes Data Plot with Input Array 
    inpt = ['11R_1789sec_tot.fits']
    for i in range(0, len(inpt)):
        new = start(filename = inpt[i])
        name = inpt[i][0:3]
        data_plot(data = new[1], fax = new[2], tag = name)
    '''
    new = start(filename = '11A_16sec.calib.4p')
    #count = 1
    #av = []
    comp1 = []
    comp2 = []
    comp3 = []
    comp4 = []
    for i in range(0, len(new[1])):
        #if i == 0:
            #av = new[1][i]
        #else:
            #av += new[1][i]
            #count += 1
    #newav = av/(count)
        new1 = fit(burst = new[1][i], mode = 'gaussian', freq = new[2][i], tag = '11A', plot = False)
        for j in range(0, len(new1[1])):
            if 350 < new1[1][j] < 362:
                #x = burst_prop(burst = new[1][i][350:362])
                comp1.append(new1[1][j])
            elif 363 < new1[1][j] < 370:
                #x = burst_prop(burst = new[1][i][363:379])
                comp2.append(new1[1][j])
            elif 380 < new1[1][j] < 390:
                #x = burst_prop(burst = new[1][i][371:390])
                comp3.append(new1[1][j])
            elif 395 < new1[1][j] < 420:
                #x = burst_prop(burst = new[1][i][395:420])
                comp4.append(new1[1][j])
        if (len(comp1) - 1) < i:
            comp1.append(np.nan)
        if (len(comp2) - 1) < i:
            comp2.append(np.nan)
        if (len(comp3) - 1) < i:
            comp3.append(np.nan)
        if (len(comp4) - 1) < i:
            comp4.append(np.nan)
    mus = [comp1, comp2, comp3, comp4]
    pplot(param = mus, pname = 'Center', fax = new[2], tag = '11A')
    '''
    new = start(filename = '11A_16sec.calib.4p')
    new1 = find_peak(data = new[1])
    sp = p.SinglePulse(data = new1[1], windowsize = 256)
    compfit = sp.component_fitting(full = True)
    print(compfit)
    '''
    test

main()
