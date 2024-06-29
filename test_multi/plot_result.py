# -------------------------------------------------------------
# File: test_multi\plot_result.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: plot result for Figure 12
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: MGT after data standardization (input of neural network); MGT without data standardization; TMI
# Output: Plot of TMI, MGT, ASz, NSS, TIlt, ThetaZ, THDR, Predicted Mask
# -------------------------------------------------------------
import torch
from utils.data_loading import MADDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from MADNet.MADNet import MADNet
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import MultipleLocator
from utils.utils import interpolate
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

major_locator = MultipleLocator(50)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 15})
binary_cmap = ListedColormap(['bisque', 'red'])
dir_data = './data_nor/'
dir_mask = './mask/'

model = MADNet(out_ch=1, in_ch=5)
model.load_state_dict(torch.load('../weights/weights.pth'))
model.eval()
dataset = MADDataset(dir_data, dir_mask)
data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

# 遍历 DataLoader 进行预测
PA = []
F1 = []
IoU = []
for index, batch in enumerate(data_loader):
    images = batch['image']
    input_tensor = images.clone().detach()
    input = input_tensor.squeeze().numpy()  # five components of MGT after data standardization

    with torch.no_grad():
        output_tensor = F.sigmoid(model(input_tensor))
    output = output_tensor.squeeze().numpy()
    output_binary = (output > 0.5).astype(np.int8)

    # get TMI
    tmi_data = pd.read_csv(f'./csv/case{index}/deltaT.csv', sep=',', header=None).to_numpy()
    tmi = tmi_data[:, 2].reshape((600, 600))
    tmi = interpolate(tmi, 100, 50)

    input = np.load(f'./data/case{index}.npy') # original MGT




    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for i in range(2):
        for j in range(6):
            ax = axes[i, j]
            axes[i, j].xaxis.set_major_locator(major_locator)
            axes[i, j].yaxis.set_major_locator(major_locator)
            axes[i, j].set_aspect('equal')


    # TMI
    im1=axes[0,0].imshow(np.flipud(tmi), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,0].set_title('TMI', fontsize=15)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Bxx
    im2=axes[0,1].imshow(np.flipud(input[0,:,:]), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,1].set_title('Bxx', fontsize=15)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Bxy
    im3=axes[0,2].imshow(np.flipud(input[1,:,:]), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,2].set_title('Bxy', fontsize=15)
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im3, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Bxz
    im4=axes[0,3].imshow(np.flipud(input[2,:,:]), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,3].set_title('Bxz', fontsize=15)
    divider = make_axes_locatable(axes[0,3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im4, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Byy
    im5=axes[0,4].imshow(np.flipud(input[3,:,:]), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,4].set_title('Byy', fontsize=15)
    divider = make_axes_locatable(axes[0,4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im5, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Byz
    im6=axes[0,5].imshow(np.flipud(input[4,:,:]), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[0,5].set_title('Byz', fontsize=15)
    divider = make_axes_locatable(axes[0,5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im6, cax=cax)
    cbar.ax.tick_params(labelsize=12)


    Bzz = 0-input[0,:,:]-input[3,:,:]
    # get comparison from traditional filters
    ASz = np.sqrt(input[2, :, :] ** 2 + input[4, :, :] ** 2 + Bzz **2)
    tilt = np.arctan(Bzz / np.sqrt(input[2, :, :] ** 2 + input[4, :, :] ** 2))
    thetaZ = np.sqrt(input[2, :, :] ** 2 + input[4, :, :] ** 2) / ASz
    B = np.empty((100, 100), dtype=object)
    nss = np.empty((100, 100), dtype=np.float64)
    for i in range(100):
        for j in range(100):
            matrix = [input[0, i, j], input[1, i, j], input[2, i, j],
                      input[1, i, j], input[3, i, j], input[4, i, j],
                      input[2, i, j], input[4, i, j], Bzz[i, j]
                      ]
            matrix = np.array(matrix, dtype=np.float64).reshape(3,3)
            eigenvalues = np.sort(np.linalg.eigvals(matrix))[::-1]
            nss[i,j] = np.sqrt(np.abs(-eigenvalues[1]**2 - eigenvalues[0]*eigenvalues[2]))
    # THDR
    dx, dy = 1, 1
    df_dx = (tmi[:, 2:] - tmi[:, :-2]) / (2*dx)
    df_dy = (tmi[2:, :] - tmi[:-2, :]) / (2*dx)
    thdr = np.sqrt(df_dx[1:-1,:]**2 + df_dy[:,1:-1]**2)

    im7=axes[1,0].imshow(np.flipud(ASz), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[1,0].set_title('ASz', fontsize=15)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im7, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    im8=axes[1,1].imshow(np.flipud(nss), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[1,1].set_title('NSS', fontsize=15)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im8, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    im9=axes[1,2].imshow(np.flipud(tilt), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[1,2].set_title('Tilt', fontsize=15)
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im9, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    im10=axes[1,3].imshow(np.flipud(thetaZ), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[1,3].set_title('ThetaZ', fontsize=15)
    divider = make_axes_locatable(axes[1,3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im10, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    im11=axes[1,4].imshow(np.flipud(thdr), extent=(-50, 50, -50, 50), cmap='rainbow')
    axes[1,4].set_title('THDR', fontsize=15)
    divider = make_axes_locatable(axes[1,4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im11, cax=cax)
    cbar.ax.tick_params(labelsize=12)


    im12=axes[1,5].imshow(np.flipud(output_binary), extent=(-50, 50, -50, 50), cmap=binary_cmap)
    axes[1,5].set_title('Predict', fontsize=15)



    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    # plt.savefig(f'./fig{index}.png', bbox_inches='tight', dpi=500)
    plt.show()