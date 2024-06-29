# -------------------------------------------------------------
# File: test_bishop\readGrd.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: Read grid data from GXF file
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: Original .grd file from Bishop model
# Output: .gxf file for further processing
# -------------------------------------------------------------
import numpy as np
import re
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm

colors = ["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"]
binary_cmap = ListedColormap(colors)
bounds = [500, 1500, 2500, 4000, 6500, 9000]
norm = BoundaryNorm(bounds, binary_cmap.N)

major_locator = MultipleLocator(100)
incl = [30, 45, 60]
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})
for i in range(1):
    dataset = gdal.Open('./gxf/bishop5x_susceptibility.gxf')
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    array = array[0:1901, :]
    array = array[100:1800,100:1800]
    rows, cols = array.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # np.savetxt(f'./csv/basement.csv', array, delimiter=',')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    contour = ax.contourf(x * 200 / 1000, y * 200 / 1000, array, levels=bounds, cmap=binary_cmap, norm=norm)
    # cax = add_right_cax(ax, pad=0.02*1000, width=0.02*1000)
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    plt.gca().set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(contour)
    cbar.set_label('$10^{-3} \\, SI$', fontdict={'family': 'Times New Roman', 'size':20})
    plt.xlabel('x(km)')
    plt.ylabel('y(km)')
    # plt.show()
    plt.savefig('./susceptibility_bar.png', dpi=500, bbox_inches='tight')
