import librosa
import numpy as np
import torch
from scipy.signal import welch
import matplotlib.pyplot as plt
from WaveConv.sinc_model import audio_classification

# Load and process waveform data
def load_waveform(path):
    data = np.load(path)
    return torch.tensor(data['waveform']).unsqueeze(0).unsqueeze(0)

# Function to perform forward pass and get ASB output
def forward_pass(model, data):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation
        return model.asb(data)  # Get ASB output

PATH1 = "../npz_data/noise/deepship/-15/deep_0_1_10_0.npz" #deep_0_1_100_1  deep_0_1_1000_0
# Load waveforms
soundData1 = load_waveform(PATH1)
# Load the model
model_path1 = "../model_pkl/deepship/Cut_deepShip_my_new_15.pkl"
model1 = audio_classification(16000,num_classes=4)
model1.load_state_dict(torch.load(model_path1))
model1.eval()
PATH2 = "../npz_data/noise/deepship/-10/deep_0_1_100_2.npz"
soundData2 = load_waveform(PATH2)
model_path2 = "../model_pkl/deepship/Cut_deepShip_my_new.pkl"
model2 = audio_classification(16000,num_classes=4)
model2.load_state_dict(torch.load(model_path2))
model2.eval()
# Convert data to numpy arrays
x1_before = soundData1.squeeze().numpy()
# Run forward passes for both sound datasets
x_after_asb1 = forward_pass(model1, soundData1)
x_after_asb1 = x_after_asb1.squeeze().numpy()

x2_before = soundData2.squeeze().numpy()
x_after_asb2 = forward_pass(model2, soundData2)
x_after_asb2 = x_after_asb2.squeeze().numpy()
fs = 16000
# Calculate PSDs using Welch method
frequencies1_before, psd1_before = welch(x1_before, fs, nperseg=1024)
frequencies1, psd1 = welch(x_after_asb1, fs, nperseg=1024)

frequencies2_before, psd2_before = welch(x2_before, fs, nperseg=1024)
frequencies2, psd2 = welch(x_after_asb2, fs, nperseg=1024)

plt.figure(figsize=(6,4))
plt.plot(frequencies2_before, psd2_before)
plt.title("PSD of original deepship data at -10dB")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0,None)
plt.ylim(0,None)
plt.savefig("PSD_original_10.pdf")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(frequencies2, psd2)
plt.title("PSD of enhanced deepship data at -10dB")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0,None)
plt.ylim(0,None)
plt.savefig("PSD_enhanced_10.pdf")
plt.show()

# 如果你想获得完整的前向过程并同时获取中间输出，可以修改 forward 函数
# class audio_classification_with_intermediate_output(audio_classification):
#     def forward(self, x):
#         x = self.asb(x)  # 32,1,16000
#         print("Output after ASB:", x.shape)  # 打印 ASB 后的输出
#         x = self.sinconv(x)
#         x = self.flatten(self.avgPool(x))
#         x = self.fc1(x)
#         return x
#
# # 创建新模型并加载权重
# model_with_intermediate = audio_classification_with_intermediate_output()
# model_with_intermediate.load_state_dict(torch.load('../model_pkl/deepship/Cut_deepShip_my_new_15.pkl'))
# model_with_intermediate.eval()
#
# # 获取中间层输出
# with torch.no_grad():
#     output = model_with_intermediate(x_input)
#     print(output.shape)
