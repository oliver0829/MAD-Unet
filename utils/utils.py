import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate(data, old_range, new_range):
    # 假设你的场值数组为 field_array，形状为 (600, 600)
    field_array = data

    new_x_range = [-new_range, new_range]  # 新 x 范围为 -50m 到 50m
    new_y_range = [-new_range, new_range]  # 新 y 范围为 -50m 到 50m

    # 生成新的坐标轴
    new_x = np.linspace(new_x_range[0], new_x_range[1], 100)
    new_y = np.linspace(new_y_range[0], new_y_range[1], 100)

    # 创建 RegularGridInterpolator 对象
    interpolator = RegularGridInterpolator((np.linspace(-old_range, old_range, field_array.shape[0]),
                                            np.linspace(-old_range, old_range, field_array.shape[1])),
                                           field_array)

    # 在新坐标上进行插值
    new_x_grid, new_y_grid = np.meshgrid(new_x, new_y, indexing='ij')  # 用于生成新网格
    new_field_array = interpolator((new_x_grid, new_y_grid))
    return new_field_array


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def old_methods(G):
    Gxx = G[0, :, :]
    Gxy = G[1, :, :]
    Gxz = G[2, :, :]
    Gyy = G[3, :, :]
    Gyz = G[4, :, :]



