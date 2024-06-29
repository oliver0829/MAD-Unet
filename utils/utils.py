# -------------------------------------------------------------
# File: utils.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: Utilities
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: /
# Output: /
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate(data, old_range, new_range):
    field_array = data
    new_x_range = [-new_range, new_range]
    new_y_range = [-new_range, new_range]
    new_x = np.linspace(new_x_range[0], new_x_range[1], 100)
    new_y = np.linspace(new_y_range[0], new_y_range[1], 100)
    interpolator = RegularGridInterpolator((np.linspace(-old_range, old_range, field_array.shape[0]),
                                            np.linspace(-old_range, old_range, field_array.shape[1])),
                                           field_array)
    new_x_grid, new_y_grid = np.meshgrid(new_x, new_y, indexing='ij')
    new_field_array = interpolator((new_x_grid, new_y_grid))
    return new_field_array




