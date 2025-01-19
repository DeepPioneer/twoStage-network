import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch

plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 读取音频文件
file_path = '../data/rain.wav'
data,sampling_rate = librosa.load(file_path,sr=16000)
frequencies, psd = welch(data, sampling_rate, nperseg=1024)

plt.plot(frequencies, psd)
plt.title("Power Spectral Density (PSD)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0,None)
plt.ylim(0,None)
plt.savefig("rain_psd.pdf")

plt.show()

# fishboat vmin=5, vmax=20

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data,n_fft=1024,hop_length=512))),
                         sr=sampling_rate,hop_length=512,x_axis="time",y_axis="hz",
                         cmap="jet",vmin=-25, vmax=5)
plt.title("Rain spectrogram")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.savefig("rain_spec.pdf")
plt.show()





