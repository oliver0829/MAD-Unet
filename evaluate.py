import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import MultipleLocator

from utils.data_loading import MADDataset
from MADNet.MADNet import MADNet
from utils.dice_score import dice_coeff

# PLOT PARAMETERS
major_locator = MultipleLocator(50)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 10})
binary_cmap = ListedColormap(['bisque', 'red'])


def evaluate(dir_data, dir_mask, dir_info):
    info = np.load(dir_info)
    model = MADNet(out_ch=1, in_ch=5)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    dataset = MADDataset(dir_data, dir_mask)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    # 遍历 DataLoader 进行预测
    PA = []
    F1 = []
    IoU = []
    for index, batch in enumerate(data_loader):
        images, true_masks = batch['image'], batch['mask']
        input_tensor = images.clone().detach()
        input = input_tensor.squeeze().numpy()  # five components of MGT
        with torch.no_grad():
            output_tensor = F.sigmoid(model(input_tensor))
        output = output_tensor.squeeze().numpy()
        mask = true_masks.squeeze().numpy()
        Bzz = 0 - input[0, :, :] - input[3, :, :]
        output_binary = (output > 0.5).astype(np.int8)

        acc = np.sum(output_binary == mask) / mask.size
        f1 = dice_coeff(output_tensor.squeeze(1), true_masks.float())
        iou = np.sum(mask * output) / np.sum(np.logical_or(mask, output_binary))

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.flipud(Bzz), extent=(-50, 50, -50, 50), cmap='rainbow')
        ax1.set_title('Bzz', fontsize=10)
        ax1.set_aspect('equal')
        ax1.xaxis.set_major_locator(major_locator)
        ax1.yaxis.set_major_locator(major_locator)

        ax2 = fig.add_subplot(132)
        ax2.imshow(np.flipud(output_binary), extent=(-50, 50, -50, 50), cmap=binary_cmap)
        ax2.set_aspect('equal')
        ax2.set_title('Segmentation result', fontsize=10)
        ax2.xaxis.set_major_locator(major_locator)
        ax2.yaxis.set_major_locator(major_locator)

        ax3 = fig.add_subplot(133)
        ax3.imshow(np.flipud(mask), extent=(-50, 50, -50, 50), cmap=binary_cmap)
        ax3.set_aspect('equal')
        ax3.set_title('True mask', fontsize=10)
        ax3.xaxis.set_major_locator(major_locator)
        ax3.yaxis.set_major_locator(major_locator)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.text(0.05, 0.9, f'Acc: {acc:.8f}', fontsize=10)
        fig.text(0.05, 0.85, f'F1: {f1:.8f}', fontsize=10)
        fig.text(0.05, 0.8, f'IoU: {iou:.8f}', fontsize=10)
        fig.text(0.05, 0.7, f'{info[index]}', fontsize=14)
        plt.show()


if __name__ == '__main__':
    # different inclination/declination angles
    dir_data = './test_data/single_object/incl_decl/data_nor/'
    dir_mask = './test_data/single_object/incl_decl/mask/'
    dir_info = './test_data/single_object/incl_decl/info.npy'
    evaluate(dir_data, dir_mask, dir_info)

    # complex object geometries
    dir_data = './test_data/single_object/complex_geo/data_nor/'
    dir_mask = './test_data/single_object/complex_geo/mask/'
    dir_info = './test_data/single_object/complex_geo/info.npy'
    evaluate(dir_data, dir_mask, dir_info)

    # growing buried depths
    dir_data = './test_data/single_object/buried_dep/data_nor/'
    dir_mask = './test_data/single_object/buried_dep/mask/'
    dir_info = './test_data/single_object/buried_dep/info.npy'
    evaluate(dir_data, dir_mask, dir_info)

    # Multiple objects
    dir_data = './test_data/multiple_object/data_nor/'
    dir_mask = './test_data/multiple_object/mask/'
    dir_info = './test_data/multiple_object/info.npy'
    evaluate(dir_data, dir_mask, dir_info)

    # Noise-corrupted data
    dir_data = './test_data/single_object/noise/data_nor/'
    dir_mask = './test_data/single_object/noise/mask/'
    dir_info = './test_data/single_object/noise/info.npy'
    evaluate(dir_data, dir_mask, dir_info)
    print('end')
