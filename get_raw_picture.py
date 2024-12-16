import scipy.io as sio
import matplotlib.pyplot as plt

# 读取.mat文件
mat_data = sio.loadmat('/home/data2/cmx/HSI_LIDAR/data_begin/Houston2013/LiDAR_1.mat')

# 获取三个键的数据
# hsi_data = mat_data['HSI'].mean(-1)
lidar_data = mat_data['LiDAR']
# label_data = mat_data['label']

# 可视化并保存为PNG文件
plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.imshow(hsi_data)
# # plt.title('HSI')
# plt.axis('off')
# plt.savefig('hsi.png', dpi=300, bbox_inches='tight')

plt.subplot(1, 3, 1)
plt.imshow(lidar_data)
# plt.title('LiDAR')
plt.axis('off')
plt.savefig('/home/caomingxiang/mm_vit_code/visual_result_clip/Houston2013_lidar.png', dpi=300, bbox_inches='tight')

# plt.subplot(1, 3, 1)
# plt.imshow(label_data)
# plt.title('Label')
# plt.axis('off')
# plt.savefig('label.png', dpi=300, bbox_inches='tight')

plt.show()