import matplotlib.pyplot as plt
import librosa.display
import numpy as np

np.random.seed(123)

# 经验上来说，归一化就是让不同维度之间的特征在数值上有一定比较性，可以大大提高分类器的准确性。
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def SNR_cal(signal,noise):
    # 计算信号的平均功率和噪声的平均功率
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    # 计算信噪比
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def min_max_function(data):
    y = (data-np.min(data))/(np.max(data)-np.min(data))
    return y

noise_files = "../data/noise.wav"
original_path = "../data/clean.wav"

soundData,fs = librosa.load(original_path,sr=16000) # 降采样到16000
soundData = soundData[:fs]

noise_signal, noise_sr = librosa.load(noise_files, sr=16000)
noise_signal = noise_signal[:noise_sr]

# 如果噪声采样率与目标信号不同，则进行重采样
if fs != noise_sr:
    noise_signal = librosa.resample(noise_signal, noise_sr, fs)
# 确保噪声长度至少与目标信号一样长
if len(noise_signal) < len(soundData):
    # 如果噪声过短，则循环噪声以匹配长度
    noise_signal = np.tile(noise_signal, int(np.ceil(len(soundData) / len(noise_signal))))[
                   :len(soundData)]
# 截断噪声以匹配目标信号长度
    noise_signal = noise_signal[:len(soundData)]

# 叠加原始噪声到目标信号
mixed_signal = soundData + noise_signal

def check_snr(signal,noise):
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal,2))# 0.5722037
    noise_power = (1/noise.shape[0])*np.sum(np.power(noise,2)) # 0.90688
    SNR = 10*np.log10(signal_power/noise_power)
    return SNR

def add_fixed_snr_noise(original_signal, noise_signal, target_snr_dB):
    # 计算原始信号的功率
    signal_power = np.sum(original_signal ** 2) / len(original_signal)
    # 计算噪声信号的功率
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)
    # 计算所需的噪声功率
    target_noise_power = signal_power / (10 ** (target_snr_dB / 10))
    # 调整噪声的功率以匹配所需的信噪比
    adjusted_noise_signal = np.sqrt(target_noise_power / noise_power) * noise_signal
    snr_ad = check_snr(original_signal,adjusted_noise_signal)
    # print("噪声信噪比",snr_ad)

    # 添加噪声到原始信号
    noisy_signal = original_signal + adjusted_noise_signal

    return noisy_signal

# 添加固定信噪比的环境噪声 设置所需的信噪比
target_snr_dB = -10

# 添加固定信噪比的噪声信号到原始信号
noisy_signal = add_fixed_snr_noise(soundData, noise_signal, target_snr_dB)
import scipy.stats
def compute_statistics(signal):
    statistics = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'skew': scipy.stats.skew(signal),
        'kurtosis': scipy.stats.kurtosis(signal)
    }
    return statistics

# Example usage
orig_stats = compute_statistics(soundData)
noisy_stats = compute_statistics(noisy_signal)
print(f"Original Signal Statistics: {orig_stats}")
print(f"Noisy Signal Statistics: {noisy_stats}")

def calculate_correlation(original, noisy):
    correlation = np.corrcoef(original, noisy)[0, 1]
    return correlation

# Example usage
correlation = calculate_correlation(soundData, noisy_signal)
print(f"Correlation between Original and Noisy Signal: {correlation:.2f}")

# 计算原始信噪比
snr_cal = SNR_cal(soundData,noise_signal)
print("原始信噪比",snr_cal)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
spectrogram_abs = np.abs(librosa.stft(soundData,n_fft=1024,hop_length=512,center=False))  # 转换到对数刻度
librosa.display.specshow(librosa.amplitude_to_db(spectrogram_abs),sr=fs,hop_length=512,x_axis="time",y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("Raw audio spectrogram",fontsize=14)
plt.subplot(2,2,2)
spectrogram_noiseabs = np.abs(librosa.stft(noise_signal,n_fft=1024,hop_length=512,center=False))  # 转换到对数刻度
librosa.display.specshow(librosa.amplitude_to_db(spectrogram_noiseabs),sr=fs,hop_length=512,x_axis="time",y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("Raw noise spectrogram",fontsize=14)
plt.subplot(2,2,3)
spectrogram_mixabs = np.abs(librosa.stft(mixed_signal,n_fft=1024,hop_length=512,center=False))  # 转换到对数刻度
librosa.display.specshow(librosa.amplitude_to_db(spectrogram_mixabs),sr=fs,hop_length=512,x_axis="time",y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of 9 dB noise added to raw signal",fontsize=14)
plt.subplot(2,2,4)
spectrogram_mix = np.abs(librosa.stft(noisy_signal,n_fft=1024,hop_length=512,center=False))  # 转换到对数刻度
librosa.display.specshow(librosa.amplitude_to_db(spectrogram_mix),sr=fs,hop_length=512,x_axis="time",y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of -10 dB noise added to raw signal",fontsize=14)
plt.tight_layout()
plt.savefig('noise.pdf', dpi=300)
plt.show()




