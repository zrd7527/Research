import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall
from scipy.signal import detrend

def data_plot():
    fb = Waterfall('spliced_guppi_57991_51723_DIAG_FRB121102_0012.gpuspec.0001.8.fil', t_start = 0, t_stop = 1000)
    fb.info()
    freqs, data = fb.grab_data(4000, 8000, 1)
    newdat = []
    for i in range(0, len(data[0])):
        newarr = []
        for j in range(0, len(data)):
            newarr.append(data[j][i])
        newdat.append(newarr)
    clean = detrend(newdat)
    plt.imshow(clean, origin = 'lower', interpolation = 'nearest', aspect = 'auto')
    plt.savefig('DetrendedData_51723')

def main():
    data_plot()
main()
