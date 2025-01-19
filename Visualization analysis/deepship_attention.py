import torch
import numpy as np
import matplotlib.pyplot as plt
from WaveConv.sinc_model import audio_classification

# Load the model
model_path = "../model_pkl/deepship/Cut_deepShip_my_new.pkl"

model = audio_classification(16000,num_classes=4)
model.load_state_dict(torch.load(model_path))
model.eval()
def load_waveform(path):
    data = np.load(path)
    return torch.tensor(data['waveform']).unsqueeze(0).unsqueeze(0)

# Function to perform forward pass and get ASB output
def forward_pass(model, data):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation
        return model.asb(data)  # Get ASB output

PATH = "../npz_data/noise/deepship/-10/deep_0_1_1_2.npz"
soundData = load_waveform(PATH)
print(soundData.shape)
output = model(soundData)  # 进行前向传播
print(output.shape)
# 定义一个全局变量来存储特征
features = []

# 定义钩子函数
def hook_fn(module, input, output):
    features.append(output.detach())

# 注册钩子
handle = model.RecModel.conv5_x.register_forward_hook(hook_fn)

# 前向传播
with torch.no_grad():
    output = model(soundData)

# 获取特征
conv5_x_features = features[0]

# 可视化特征
def visualize_features(features, num_channels=6):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            # 获取第 i 个通道的特征
            channel_features = features[0, i, :].cpu().numpy()
            # 绘制特征图
            ax.imshow(channel_features.reshape(1, -1), origin='lower', aspect='auto')
            ax.axis('off')
        else:
            ax.axis('off')
    # axes.set_title("Convolutional layer feature map with Multi-Spectral Attention")
    plt.savefig("deepship_conv5.pdf")
    plt.show()

# 可视化 conv5_x 的特征
visualize_features(conv5_x_features, num_channels=8)

# 移除钩子
handle.remove()

# import torch
# import matplotlib.pyplot as plt
#
#
# # 定义可视化函数
# def visualize_multi_spectral_attention(features, num_channels=6):
#     """
#     可视化 MultiSpectral Attention Layer 输出的特征图。
#
#     features: 从卷积层得到的特征图
#     num_channels: 可视化的通道数量
#     """
#     # 创建一个 2x3 的子图
#     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#     for i, ax in enumerate(axes.flat):
#         if i < num_channels:
#             # 获取第 i 个通道的特征
#             channel_features = features[0, i, :].cpu().numpy()
#             # 绘制特征图
#             ax.imshow(channel_features.reshape(1, -1), origin='lower', aspect='auto')
#             ax.axis('off')  # 不显示坐标轴
#         else:
#             ax.axis('off')  # 超出通道数时不显示
#     plt.show()
#
#
# # 可视化包含 MultiSpectral Attention Layer 的 conv5_x 特征
# visualize_multi_spectral_attention(conv5_x_features, num_channels=6)
