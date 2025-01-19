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

PATH1 = "../npz_data/noise/ShipEar/-15/10__10_07_13_100_2.npz"
PATH2 = "../npz_data/noise/ShipEar/-10/10__10_07_13_100_2.npz"

# Load waveforms
soundData1 = load_waveform(PATH1)
soundData2 = load_waveform(PATH2)

# Load the model
model_path1 = "../model_pkl/ShipEar/Cut_ShipEar_my_new_15.pkl"
model_path2 = "../model_pkl/ShipEar/Cut_ShipEar_my_new.pkl"
model1 = audio_classification(16000,num_classes=5)
model1.load_state_dict(torch.load(model_path1))

model2 = audio_classification(16000,num_classes=5)
model2.load_state_dict(torch.load(model_path2))

# Run forward passes for both sound datasets
x_after_asb1 = forward_pass(model1, soundData1)
x_after_asb2 = forward_pass(model2, soundData2)

# Convert data to numpy arrays
x1_before = soundData1.squeeze().numpy()
x_after_asb1 = x_after_asb1.squeeze().numpy()
x2_before = soundData2.squeeze().numpy()
x_after_asb2 = x_after_asb2.squeeze().numpy()

# Calculate PSDs using Welch method
frequencies1_before, psd1_before = welch(x1_before, 16000, nperseg=1024)
frequencies1, psd1 = welch(x_after_asb1, 16000, nperseg=1024)

frequencies2_before, psd2_before = welch(x2_before, 16000, nperseg=1024)
frequencies2, psd2 = welch(x_after_asb2, 16000, nperseg=1024)


plt.figure(figsize=(6,4))
plt.plot(frequencies2_before, psd2_before)
plt.title("PSD of original ShipsEar data at -10dB")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0,None)
plt.ylim(0,None)
plt.savefig("PSD_originalShipsEar_10.pdf")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(frequencies2, psd2)
plt.title("PSD of enhanced ShipsEar data at -10dB")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0,None)
plt.ylim(0,None)
plt.savefig("PSD_enhancedShipsEar_10.pdf")
plt.show()
