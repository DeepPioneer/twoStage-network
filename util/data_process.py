import librosa
import numpy as np
import glob,os

# 原始音频文件
def preprocess_and_save(file_path, save_dir,CLASS_MAPPING):
    # 创建一个字典，将类别映射为整数
    category_to_int = {category: idx for idx, category in enumerate(CLASS_MAPPING)}

    files = [os.path.join(file_path, x) for x in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, x))]

    for idx, folder in enumerate(files):
        for audio_path in glob.glob(folder + '/*.wav'):
            soundData, fs = librosa.load(audio_path, sr=16000)  # 降采样到16000
            # 文件名处理
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            label = os.path.basename(folder)
            label_int = category_to_int.get(label, -1)
            npz_path = os.path.join(save_dir, f"{base_name}_{label_int}.npz")
            # 保存为npz文件
            np.savez(npz_path, waveform=soundData, label=label_int)

if __name__ == "__main__":
    
    # 目标数据
    target_file_path = "../../dataset/Cut_whale"  # "../audio_dataset/ESC" Cut_ShipEar  Cut_deepShip Cut_whale
    data_type = os.path.basename(target_file_path)
    
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

    save_dir = r"../save_npz/{}".format(data_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocess_and_save(target_file_path, save_dir,CLASS_MAPPING)
    print("data loading fininshing")

