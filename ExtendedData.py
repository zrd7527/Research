import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall

def main():
    print('Hello World')
    fb = Waterfall('spliced_guppi_57991_51723_DIAG_FRB121102_0012.gpuspec.0001.8.fil', t_start = 0, t_stop = 1000)
    fb.info()
    freqs, data = fb.grab_data(4000, 8000, 1)
    plt.imshow(data, origin = 'lower', interpolation = 'nearest', aspect = 'auto')
    plt.savefig('NewData_57991')
main()
