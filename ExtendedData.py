import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall
print('Hello World')
fb = Waterfall('blc20_guppi_57991_49836_DIAG_FRB121102_0010.0002.raw', t_start = 0, t_stop = 1000)
fb.info()
