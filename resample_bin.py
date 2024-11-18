import numpy as np
from scipy.signal import resample

# 假设你已经有了 16000 Hz 的原始音频数据，存储在一个二进制文件中
# 首先，我们需要从文件中读取这些二进制数据
def read_binary_data(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    # 假设数据是以 16 位 PCM 格式存储的
    return np.frombuffer(data, dtype=np.int16)

# 将数据重采样
def resample_audio(data, original_rate, target_rate):
    # 计算新的样本数量
    num_samples = int(len(data) * target_rate / original_rate)
    # 使用 scipy 的 resample 函数进行重采样
    resampled_data = resample(data, num_samples)
    return resampled_data.astype(np.int16)  # 确保数据类型为 int16

# 将重采样后的数据保存为二进制文件
def write_binary_data(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(data.tobytes())

# 主程序
def main():
    original_rate = 8000
    target_rate = 16000
    duration = 2  # 时长为 2 秒

    # 读取原始音频数据
    input_file = 'test_data/nihaosuanneng.bin'  # 输入文件路径
    audio_data = read_binary_data(input_file)

    # 重采样
    resampled_data = resample_audio(audio_data, original_rate, target_rate)

    # 保存重采样后的数据
    output_file = 'output_audio.bin'  # 输出文件路径
    write_binary_data(output_file, resampled_data)

    print(f"成功将 {input_file} 从 {original_rate} Hz 重采样到 {target_rate} Hz，输出文件为 {output_file}")

if __name__ == "__main__":
    main()
