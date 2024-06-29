# -------------------------------------------------------------
# File: test_bishop\computeMGTfromTMI.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: Compute MGT from TMI using Fourier processing
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: TMI
# Output: data\case{}.npy(MGT without data standardization)
# -------------------------------------------------------------

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.constants import mu_0
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def interpolate(data):
    min_length = min(data.shape[1], data.shape[2])
    A_cropped = data[:, :min_length, :min_length]
    x = np.linspace(0, 1, A_cropped.shape[1])
    y = np.linspace(0, 1, A_cropped.shape[2])
    x_new = np.linspace(0, 1, 100)
    y_new = np.linspace(0, 1, 100)
    A_new = np.empty((5, 100, 100))
    for i in range(data.shape[0]):
        interpolating_function = RegularGridInterpolator((x, y), A_cropped[i], bounds_error=False, fill_value=None)
        x_new_grid, y_new_grid = np.meshgrid(x_new, y_new, indexing='ij')
        points = np.array([x_new_grid.flatten(), y_new_grid.flatten()]).T
        A_new[i] = interpolating_function(points).reshape((100, 100))
    return A_new


def FourierTransform(tmi_data, l, m, n,):
    # From https://doi.org/10.1016/j.jappgeo.2016.08.010
    ny, nx = tmi_data.shape
    u = np.fft.fftfreq(nx)
    v = np.fft.fftfreq(ny)
    kx, ky = np.meshgrid(u, v)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    k = np.sqrt(kx ** 2 + ky ** 2)

    delta_T = tmi_data.copy()
    delta_T_fft = fft2(delta_T)


    denominator = k * n + 1j * (kx * l + ky * m)
    epsilon = 1e-20
    denominator[np.abs(denominator) < epsilon] = epsilon

    bxx_fft = -kx ** 2 / denominator * delta_T_fft
    bxy_fft = -kx * ky / denominator * delta_T_fft
    bxz_fft = 1j * kx * k / denominator * delta_T_fft

    byy_fft = -ky ** 2 / denominator * delta_T_fft
    byz_fft = 1j * ky * k / denominator * delta_T_fft


    bxx = np.real(ifft2(bxx_fft))
    bxy = np.real(ifft2(bxy_fft))
    bxz = np.real(ifft2(bxz_fft))

    byy = np.real(ifft2(byy_fft))
    byz = np.real(ifft2(byz_fft))

    MGT = np.stack((bxx, bxy, bxz, byy, byz), axis=0)
    MGT = MGT[:,100:1800,100:1800]
    return MGT

def plot_fields(tmi_data, mgt):

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(mgt[0,:,:], cmap='jet', origin='lower')
    ax.set_title('Bxx', fontsize=20)
    # fig.colorbar(im, ax=ax)


    ax = axes[0, 1]
    im = ax.imshow(mgt[1,:,:], cmap='jet', origin='lower')
    ax.set_title('Bxy', fontsize=20)
    # fig.colorbar(im, ax=ax)


    ax = axes[0, 2]
    im = ax.imshow(mgt[2,:,:], cmap='jet', origin='lower')
    ax.set_title('Bxz', fontsize=20)
    # fig.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(tmi_data, cmap='jet', origin='lower')
    ax.set_title('TMI', fontsize=20)
    # fig.colorbar(im, ax=ax)


    ax = axes[1, 1]
    im = ax.imshow(mgt[3,:,:], cmap='jet', origin='lower')
    ax.set_title('Byy', fontsize=20)
    # fig.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.imshow(mgt[4,:,:], cmap='jet', origin='lower')
    ax.set_title('Byz', fontsize=20)
    # fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


incl = ['30', '45', '60']
for i in range(2,3):
    Incl = incl[i]
    deltaT = pd.read_csv(f'./csv/TMI_{Incl}.csv', sep=',', header=None).to_numpy()

    incl = [30, 45, 60]
    incl = incl[i]  # deg
    decl = 0  # deg
    l, m, n = (np.cos(incl*np.pi/180) * np.sin(decl*np.pi/180),
               np.cos(incl*np.pi/180) * np.cos(decl*np.pi/180),
               -np.sin(incl*np.pi/180))

    mgt= FourierTransform(deltaT*1e-9, l, m, n)



    MGT_output = np.zeros((5, 100, 100))

    MGT_output = interpolate(mgt)
    plot_fields(deltaT*1e-9, MGT_output)
    # np.save(f'./data/case{i}.npy', MGT_output)

