import librosa
import numpy as np
import glob,os
import torch
from random import random
import matplotlib.pyplot as plt
# Set random seeds for reproducibility
random_name = str(random())
random_seed = 123   
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

def check_snr(signal,noise):
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal,2))# 0.5722037
    noise_power = (1/noise.shape[0])*np.sum(np.power(noise,2)) # 0.90688
    SNR = 10*np.log10(signal_power/noise_power)
    return SNR

def add_fixed_snr_noise(original_signal, noise_signal, target_snr_dB):
    ori_snr = check_snr(original_signal,noise_signal)
    # print("原始信噪比",ori_snr)
    # 计算原始信号的功率
    signal_power = np.sum(original_signal ** 2) / len(original_signal)
    # 计算噪声信号的功率
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)
    # 计算所需的噪声功率
    target_noise_power = signal_power / (10 ** (target_snr_dB / 10))
    # 调整噪声的功率以匹配所需的信噪比
    adjusted_noise_signal = np.sqrt(target_noise_power / noise_power) * noise_signal
    after_snr = check_snr(original_signal, adjusted_noise_signal)
    # print("新的信噪比",after_snr)
    # 添加噪声到原始信号
    noisy_signal = original_signal + adjusted_noise_signal

    return noisy_signal,ori_snr,after_snr

def preprocess_and_save(target_file_path, background_file_path, save_dir,target_snr_dB,CLASS_MAPPING):
    # 创建一个字典，将类别映射为整数
    category_to_int = {category: idx for idx, category in enumerate(CLASS_MAPPING)}
    ori_snr_list = []
    after_snr_list = []
    # 获取目标噪声文件和背景噪声文件路径
    target_files = [os.path.join(target_file_path, x) for x in os.listdir(target_file_path) if
                    os.path.isdir(os.path.join(target_file_path, x))]
    # background_files = [os.path.join(background_file_path, f) for f in os.listdir(background_file_path) if
    #                     f.endswith('.wav')]
    
    background_files = [os.path.join(background_file_path, f) for f in os.listdir(background_file_path) if
                        f.startswith('81') and f.endswith('.wav')]

    for idx, folder in enumerate(target_files):
        for audio_path in glob.glob(folder + '/*.wav'):
            # 读取目标噪声数据
            target_data, fs = librosa.load(audio_path, sr=16000)  # 降采样到16000
            # 生成相同长度的高斯噪声
            background_data = np.random.normal(0, 1, 16000)
            # 随机选取一个背景噪声文件并读取数据
            # background_path = np.random.choice(background_files)
            # background_data, _ = librosa.load(background_path, sr=16000)

            # 确保背景噪声数据长度不小于目标噪声数据长度
            if len(background_data) < len(target_data):
                background_data = np.tile(background_data, int(np.ceil(len(target_data) / len(background_data))))[
                                  :len(target_data)]

            # 调整背景噪声的强度以达到所需的信噪比
            mixed_data,ori_snr,after_snr = add_fixed_snr_noise(target_data,background_data,target_snr_dB)
            ori_snr_list.append(ori_snr)
            after_snr_list.append(after_snr)
            # print(ori_snr,after_snr)

            # 文件名处理
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            label = os.path.basename(folder)
            label_int = category_to_int.get(label, -1)
            # print(label_int)
            npz_path = os.path.join(save_dir, f"{base_name}_{label_int}.npz")
            # 保存为npz文件
            np.savez(npz_path, waveform=mixed_data, label=label_int)
    return ori_snr_list, after_snr_list

if __name__ == "__main__":
    
    target_snr_dB = -15
    # 调用预处理函数
    target_file_path = "../../dataset/Cut_ShipEar"   # "../audio_dataset/ESC" Cut_ShipEar  Cut_deepShip Cut_whale
    data_type = os.path.basename(target_file_path)
    background_file_path = "../../dataset/noise"
    save_dir = r"../negativeFifteen_npz/{}".format(data_type) # noise_negativeFifteen_npz noise_negativeTen_npz noise_negativeFive_npz
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 调用预处理函数
    if data_type == "Cut_ESC":
        CLASS_MAPPING = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn']
    elif data_type == "Cut_deepShip":
        CLASS_MAPPING = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    elif data_type == "Cut_ShipEar":
        CLASS_MAPPING = ["0", "1", "2", "3", "4"]
    elif data_type == "Cut_whale":
        CLASS_MAPPING = ["Background_noise", "Bowhead_whale", "Humpback_whale", "Pilot_whale", "Ship_signal"]
    else:
        CLASS_MAPPING = None

    ori_snr_list, after_snr_list = preprocess_and_save(target_file_path, background_file_path, save_dir,target_snr_dB,CLASS_MAPPING)
    print("data process finishing!")
    # 可视化 SNR
    plt.figure(figsize=(10, 6))
    plt.plot(ori_snr_list, label='Original SNR')
    # plt.plot(after_snr_list, label='After SNR')
    plt.xlabel('File Index')
    plt.ylabel('SNR (dB)')
    plt.title('Original and After SNR for Each File')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'noise_{data_type}_{target_snr_dB}.pdf',bbox_inches='tight',pad_inches=0,dpi=300)
    # plt.show()

