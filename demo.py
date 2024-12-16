# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.io import loadmat


# if __name__ == '__main__':
#     data = loadmat('/home/data2/cmx/HSI_LIDAR/data_begin/Houston2013/HSI.mat')
#     # print(data['hsi'].shape, data['lidar'].shape)
#     hsi = data['HSI'].mean(-1)
#     # lidar = data['LiDAR'].mean(-1)
#     # label = data['label']
#     # print(label)

#     plt.figure(figsize=(12, 12))
#     plt.subplot(221)
#     plt.imshow(hsi)
#     plt.axis('off')
#     plt.savefig('/home/caomingxiang/mm_vit_code/visual_result_clip/Houston2013_HSI.png', bbox_inches='tight')


#     # plt.subplot(222)
#     # plt.imshow(lidar)

#     # plt.subplot(223)
#     # plt.imshow(label)
#     # plt.show()

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 读取.mat文件
mat_file = '/home/data2/cmx/HSI_LIDAR/data_begin/Houston2013/HSI.mat'
data = scipy.io.loadmat(mat_file)

# 假设数据存储在名为'hypercube'的变量中
hypercube = data['HSI']

# 检查数据维度
print("origin shape:", hypercube.shape)
# image = hypercube.mean(-1)
# print("new shape:", image.shape)

# 选择一个波段进行可视化，假设选择第10个波段
# band = 10
# image = hypercube[:, :, band]

# 显示图像
# plt.imshow(image, cmap='gray')
plt.imshow(hypercube)
plt.axis('off')
plt.savefig('/home/caomingxiang/mm_vit_code/visual_result_clip/Houston2013_HSI.png')
plt.close()