import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticVoiceDataset(Dataset):
    def __init__(self, root_dir, target_sr=8000, target_length=16000):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.target_length = target_length
        self.file_paths = []
        self.labels = []

        # 按类别获取所有数据文件名
        for label in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, label)
            if os.path.isdir(category_dir):
                wav_files = glob.glob(os.path.join(category_dir, '*.wav'))
                for wav_file in wav_files:
                    self.file_paths.append(wav_file)
                    self.labels.append(int(label))  

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_file = self.file_paths[idx]
        label = self.labels[idx]

        # 使用librosa加载音频，并重采样
        audio, sr = librosa.load(wav_file, sr=self.target_sr)

        if len(audio)> 16000:
            start_index = np.random.randint(0, len(audio)-16000+1)
            audio = audio[start_index: start_index+16000]
        
        # 预处理
        max_rate = 0.2
        top_n = 500
        top_values = np.partition(np.abs(audio), -top_n)[-top_n:]  
        mean_top_values = np.mean(top_values)
        audio = audio/mean_top_values*max_rate


        audio = torch.tensor(audio)

        return audio, label

