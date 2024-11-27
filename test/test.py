import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math
from math import sqrt
from kneed import KneeLocator
from numpy import  log10
from scipy import optimize
import scipy.signal as signal
import  cmath
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import impedance
import unittest
import pandas as pd
from io import StringIO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ECM_func import EIS_fit_test



class TestEISfit(unittest.TestCase):
    '''test the process_csv function'''
    def setUp(self):
        # initialize the test data
        self.EIS_data = {
    'freq': [
        20004.453, 15829.126, 12516.703, 9909.4424, 7835.48, 6217.2461, 4905.291, 3881.2737, 3070.9827,
        2430.7778, 1923.1537, 1522.4358, 1203.8446, 952.86591, 754.27557, 596.71857, 471.96338,
        373.20856, 295.47278, 233.87738, 185.05922, 146.35823, 115.77804, 91.6721, 72.51701,
        57.36816, 45.3629, 35.93134, 28.40909, 22.48202, 17.79613, 14.06813, 11.1448, 8.81772,
        6.97545, 5.5173, 4.36941, 3.45686, 2.73547, 2.16054, 1.70952, 1.35352, 1.07079, 0.84734,
        0.67072, 0.53067, 0.41976, 0.33183, 0.26261, 0.20791, 0.16452, 0.13007, 0.10309, 0.08153,
        0.06443, 0.05102, 0.04042, 0.03192, 0.02528, 0.01999
    ],
    'im': [
        -0.01529, 0.00369, 0.02209, 0.03551, 0.04718, 0.05914, 0.06986, 0.07986, 0.08863, 0.09917,
        0.10721, 0.11396, 0.12434, 0.13192, 0.14222, 0.15075, 0.16147, 0.17045, 0.18028, 0.19103,
        0.2023, 0.21429, 0.22499, 0.23517, 0.24557, 0.25348, 0.26187, 0.26869, 0.27612, 0.28389,
        0.29396, 0.30497, 0.3145, 0.32408, 0.32287, 0.32151, 0.31169, 0.29755, 0.27029, 0.24519,
        0.21807, 0.19033, 0.16595, 0.14529, 0.12909, 0.11642, 0.10679, 0.10143, 0.10122, 0.10569,
        0.11186, 0.11977, 0.12989, 0.14567, 0.16709, 0.1906, 0.21682, 0.25615, 0.31228, 0.37682
    ],
    're': [
        0.30613, 0.31339, 0.32342, 0.33392, 0.34512, 0.35809, 0.37241, 0.38672, 0.40245, 0.41866,
        0.43547, 0.45472, 0.4732, 0.49255, 0.51459, 0.5374, 0.56491, 0.58797, 0.61613, 0.6465,
        0.67678, 0.70985, 0.74454, 0.78098, 0.82043, 0.86006, 0.90016, 0.9415, 0.98055, 1.02311,
        1.06673, 1.11305, 1.16265, 1.22, 1.28077, 1.32956, 1.3946, 1.44193, 1.48662, 1.52887,
        1.55943, 1.58206, 1.60076, 1.61536, 1.6267, 1.63551, 1.64386, 1.65298, 1.66191, 1.66964,
        1.67593, 1.68287, 1.69335, 1.70755, 1.72473, 1.74406, 1.77413, 1.81946, 1.87026, 1.91763
    ]
    }
        self.EIS_data = pd.DataFrame(self.EIS_data)

    def test_process_csv(self):
        # set the column index to 1
        processed_result = EIS_fit_test(self.EIS_data)
        expected_result = [5e-324, 0.25999205231851896, 0.02852426837159198, 0.655404617059859, 1.9190583524139428, 3.0819423274744686, 84.12530877684728, 17.41176435316297, 0.9999999999999999]
        processed_result = pd.DataFrame(processed_result)
        expected_result = pd.DataFrame(expected_result)
        # check if the result is equal to the expected data
        pd.testing.assert_frame_equal(processed_result, expected_result)

if __name__ == '__main__':
    unittest.main()