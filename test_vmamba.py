import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import Dataset
import data_loader
from sklearn.metrics import *
from utils import config
from net import nets, net_v1, net_vmamba_v2, net_vmamba_v3, net_vmamba_v4

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def aa(yt, j1t):
    yt = torch.from_numpy(np.array(yt))
    j1t = torch.from_numpy(np.array(j1t))
    mem = [0] * num_class
    all_mem = [0] * num_class
    n_r = [0] * num_class
    n_p = [0] * num_class
    oa = torch.eq(j1t, yt)

    for x, y in zip(yt, j1t):
        n_r[x] += 1
        n_p[y] += 1
        all_mem[x] += 1
        if x == y:
            # print(x, y )
            mem[x] += 1
    # print(mem)
    mem = np.array(mem)
    all_mem = np.array(all_mem) + 1e-3
    n_r = np.array(n_r)
    n_p = np.array(n_p)
    pe = np.sum(n_r * n_p) / (yt.shape[0] * yt.shape[0])
    acc_per_class = mem / all_mem
    # print(mem, all_mem)
    # print(acc_per_class)
    AA = np.mean(mem / all_mem)
    oa = torch.mean(oa.float())

    k = (oa - pe) / (1 - pe)
    # print(AA)
    return oa, AA, k, acc_per_class


if __name__ == '__main__':
    device = 'cuda'
    use_data = 'MUFFL'        #'Augsburg'   'MUFFL'  'Houston2013
    config = config.configs[use_data]
    num_class = config.num_class
    # save_weights_path = './save_weights_' + use_data + '/weights.pth'
    save_weights_path = '/home/data2/cmx/save_weights_MUFFL/A_share_B_exchange_no_downsample_5ka_accsave/best_weights.pth'
    train_files, test_files = data_loader.get_data_path(use_data)

    model = net_vmamba_v2.VSSM(input_shape=(32, 32), in_chans_hsi=config.input_hsi_channel, in_chans_lidar=config.input_lidar_channel,
                               num_classes=config.num_class).to(device)

    key = model.load_state_dict(torch.load(save_weights_path, map_location=device)['model_dict'])
    print(key)
    model.eval()

    # # 示例数据集和可视化
    # dataset = Dataset(test_files)
    # dataset.visualize(0)  # 可视化第一个样本
    
    test_data_loader = DataLoader(Dataset(test_files), batch_size=1,
                                  num_workers=4, shuffle=False)
    labels, predicts = [], []
    for x_hsi, x_lidar, mask, y in tqdm(test_data_loader):
        x_hsi = x_hsi.to(device)
        x_lidar = x_lidar.to(device)
        mask = mask.numpy().flatten()
        y = y.numpy().flatten()
        idx = np.where(mask > 0)
        y = y[idx]
        labels.extend(y)

        with torch.no_grad():
            # *_, pred = model(x_hsi, x_lidar)
            pred, pred_y, pred_z = model(x_hsi, x_lidar)
        pred = torch.softmax(pred_z, dim=1)
        pred = pred.cpu().numpy()[0]
        pred = np.argmax(pred, 0).flatten()[idx]
        predicts.extend(pred)
    print(classification_report(labels, predicts))
    epoch_acc_dev, AA, k, acc_per_class = aa(labels, predicts)
    print(epoch_acc_dev.item(), AA.item(), k.item(), acc_per_class)
