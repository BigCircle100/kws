import numpy as np
from scipy.io.wavfile import write

def bin_to_wav(bin_file_path, wav_file_path, sample_rate=16000):
    # 读取二进制文件
    with open(bin_file_path, 'rb') as f:
        # 读取数据并转换为short类型
        data = np.fromfile(f, dtype=np.int16)

    # 将数据归一化到[-1.0, 1.0]范围（如果需要）
    normalized_data = data / np.max(np.abs(data))

    # 保存为wav文件
    write(wav_file_path, sample_rate, normalized_data)

# 使用示例
bin_file_path = 'output_audio.bin'  # 替换为你的bin文件路径
wav_file_path = 'output.wav'               # 输出的wav文件路径
sample_rate = 16000                         # 根据需要设置采样率

bin_to_wav(bin_file_path, wav_file_path, sample_rate)
