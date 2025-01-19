import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 文件夹路径
folder_path = '../npz_data/origin/shipsear'  # 假设DeepShip文件夹在当前工作目录下

# 假设你的 npz 文件存放在当前目录
# npz_files = ['15__10_07_13_0.wav', '21__18_07_13_1.wav', '6__10_07_13__2.wav',
#              '16__10_07_13_3.wav','81__25_09_13_4.wav']

npz_files = ['s_0.wav', 's_1.wav', 's_2.wav','s_3.wav','s_4.wav']

# 用于存储不同类别的数据
category_data = {0: [], 1: [], 2: [], 3: [], 4: []}

# 加载数据
for file in npz_files:
    file = os.path.join(folder_path, file)
    print(file)
    # 从文件名中提取类别
    category = int(file.split('_')[-1][0])  # 获取文件名中的最后一位数字作为类别
    print(category)
    data,fs = librosa.load(file,sr=16000)  # 加载npz文件
    data = data[:fs*5]
    # 假设每个文件包含一个名为 'data' 的数组（请根据实际情况修改）
    category_data[category].append(data)

fig, ax = plt.subplots(figsize=(8, 6))
# 将每个类别的所有数据合并
data_for_boxplot = [np.concatenate(category_data[i]) for i in range(5)]
box_colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow','lightcoral']#

bp = ax.boxplot(data_for_boxplot, labels=[f'Category {i}' for i in range(5)],
                sym="r+", showmeans=True)

# for patch, color in zip(bp['boxes'], box_colors):
#     patch.set_facecolor(color)

# median_colors = ['red', 'purple', 'blue', 'orange','green']
#
# for i, median in enumerate(bp['medians']):
#     median.set(color=median_colors[i], linewidth=2)

ax.set_title('Boxplot for Multi-Class ShipsEar Data',fontsize=16)
ax.set_xlabel('Category')
ax.set_ylabel('Values')
ax.grid(True, linestyle='--', alpha=0.7)
plt.savefig("shipsear.pdf")
plt.show()

