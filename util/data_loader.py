import numpy as np
import torch
import glob,os
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter

class SoundDataset(Dataset):
    def __init__(self, preprocessed_path):
        self.audio_paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
        # print(self.audio_paths)
        self.labels = [int(os.path.basename(x).split('_')[-1].replace('.npz', '')) for x in self.audio_paths]

    def __getitem__(self, index):
        npz_path = self.audio_paths[index]
        data = np.load(npz_path)
        waveform = data['waveform']
        label = data['label']
        soundData = torch.tensor(waveform)
        # 提取文件名（没有扩展名）
        # file_name = os.path.splitext(os.path.basename(npz_path))[0]
        # return soundData, label,file_name
        return soundData, label

    def __len__(self):
        return len(self.audio_paths)
    
def split_dataset(dataset, train_ratio=0.8, random_seed=123):
    np.random.seed(random_seed)  # 设置随机种子
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * total_size))

    train_indices = indices[:train_split]

    test_indices = indices[train_split:]

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    # 获取训练集和测试集的标签
    train_labels = [dataset.labels[i] for i in train_indices]
    test_labels = [dataset.labels[i] for i in test_indices]

    # 统计标签数量
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)

    return train_set, test_set, train_label_counts, test_label_counts


def get_data_loaders(preprocessed_path, batch_size, train_ratio=0.9, random_seed=123, num_workers=4):
    dataset = SoundDataset(preprocessed_path)
    train_set, test_set, train_label_counts, test_label_counts = split_dataset(dataset, train_ratio, random_seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, train_label_counts, test_label_counts

# 测试数据集
if __name__ == "__main__":
    preprocessed_path = "../save_npz/ESC"
    data_set = SoundDataset(preprocessed_path)
    print(f'Dataset length: {len(data_set)}')
    for i in range(len(data_set)):
        data, label = data_set[i]
        # print(f"Data shape: {data.shape}, Label: {label}")
    train_set, test_set = split_dataset(data_set, train_ratio=0.8, test_ratio=0.2)

    print(f'Training set length: {len(train_set)}')
    # print(f'Validation set length: {len(val_set)}')
    print(f'Test set length: {len(test_set)}')



    kf = KFold(n_splits=5, shuffle=True)
    # 进行交叉验证
    fold_losses = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_set)):
        print(f'Fold {fold + 1}/{5}')
        train_subset = torch.utils.data.Subset(data_set, train_idx)
        test_subset = torch.utils.data.Subset(data_set, test_idx)
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        print(len(train_idx),len(test_idx))
        train_labels = np.array([data_set[idx][1] for idx in train_idx])  # Assuming the second element is the label
        class_distribution = Counter(train_labels)
        print(f"Class distribution in training set for fold {fold + 1}: {class_distribution}")
        print(class_distribution[0])

        # train_labels = np.array([data_set[idx][1] for idx in test_idx])  # Assuming the second element is the label
        # class_distribution = Counter(train_labels)
        # print(f"Class distribution in test_idx set for fold {fold + 1}: {class_distribution}")

        for class_label in range(5):
            print(f"Class {class_label}: {class_distribution[class_label]}")
        class_weights = []
        for i in range(5):
            class_weights.append(class_distribution[i]/len(train_idx))
        print(class_weights)
        # for i, data in enumerate(test_loader):
        #     inputs, labels,file_name = data
        #     print(file_name,inputs.shape,labels)
