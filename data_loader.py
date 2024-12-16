import glob
from torch.utils import data
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_data_path(data_name):
    path_in = '/home/data2/cmx/HSI_LIDAR/data/' + data_name
    train_path = path_in + '/train'
    test_path = path_in + '/test'
    train_files = glob.glob(train_path + '/*.mat')
    test_files = glob.glob(test_path + '/*.mat')
    return train_files, test_files


class Dataset(data.Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, item):
        data = loadmat(self.files[item])
        hsi = data['hsi']
        lidar = data['lidar']
        label = data['label']

        ########### visualize input image   ###########
        # data = loadmat(self.files[1])
        # hsi_rgb = data['hsi'].mean(-1)
        # lidar_patch = data['lidar']
        # # 选择三个通道 (例如第3、20、40通道)来组合成类似RGB的图像
        # # hsi_rgb = hsi_patch[..., [5, 15, 20]]
        # # 将三个通道缩放到 0-1 范围内
        # hsi_rgb = (hsi_rgb - hsi_rgb.min()) / (hsi_rgb.max() - hsi_rgb.min())
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # ax[0].imshow(hsi_rgb)
        # ax[0].set_title('HSI Image (RGB Composite)')
        # ax[0].axis('off')
        
        # ax[1].imshow(lidar_patch, cmap='gray')
        # ax[1].set_title('Lidar Image')
        # ax[1].axis('off')
        
        # plt.tight_layout()
        # plt.savefig('mm_vit_code/muufl_hsi_patch0.png')
        # plt.savefig('mm_vit_code/muufl_lidar_patch0.png')
        # plt.close()
        ########### visualize input image   ###########

        mask = (label >= 0).astype(np.int8)
        label[label < 0] = 0
        if len(hsi.shape) == 2:
            hsi = hsi[..., None]
        if len(lidar.shape) == 2:
            lidar = lidar[..., None]
        # input_data = np.concatenate([hsi, lidar], axis=-1).transpose(2, 0, 1)

        hsi = torch.from_numpy(hsi.transpose(2, 0, 1)).type(torch.FloatTensor)
        lidar = torch.from_numpy(lidar.transpose(2, 0, 1)).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        label = torch.from_numpy(label).type(torch.FloatTensor).long()
        return hsi, lidar, mask, label

    def __len__(self):
        return len(self.files)
    
    def visualize(self, idx, save_path_hsi='mm_vit_code/hsi_image.png', save_path_lidar='mm_vit_code/lidar_image.png'):
        hsi, lidar, _, _ = self[idx]
        
        # 假设通道在 0 维，我们取出前两个像素维度进行可视化
        # HSI 通道可视化 (例如使用第一个通道作为灰度)
        hsi_image = hsi.cpu().numpy()
        hsi_image = hsi_image.mean(axis=0)
        # Lidar 通道可视化
        lidar_image = lidar.cpu().numpy()
        lidar_image = lidar_image.mean(axis=0)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(hsi_image, cmap='viridis')
        ax[0].set_title('HSI Image')
        ax[0].axis('off')
        
        ax[1].imshow(lidar_image, cmap='gray')
        ax[1].set_title('Lidar Image')
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path_hsi)
        plt.savefig(save_path_lidar)
        plt.close()


if __name__ == '__main__':
    x, y = get_data_path('Houston2013')
    print(x)


