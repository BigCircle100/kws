import numpy as np
from scipy.signal import resample

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

def find_differences_with_indices(array1, array2):
    differences = array1 != array2
    differing_indices = np.where(differences)[0]  # 获取不同值的索引
    differing_values1 = array1[differences]
    differing_values2 = array2[differences]
    
    return differing_indices, differing_values1, differing_values2

# 主程序
def main():
    original_rate = 8000
    target_rate = 16000
    duration = 2  # 时长为 2 秒

    # 读取原始音频数据
    input_file = '../test_data/nihaosuanneng.bin'  # 输入文件路径
    audio_data = read_binary_data(input_file)

    # 1.先重采样再合并
    # 重采样
    resampled_data1 = resample_audio(audio_data, original_rate, target_rate)
    res1 = np.concatenate((resampled_data1, resampled_data1))

    # 2.先合并再重采样
    audio_data2 = np.concatenate((audio_data, audio_data))
    res2 = resample_audio(audio_data2, original_rate, target_rate)

    tolerance = 3  # 设定一个容忍误差
    are_equal = np.all(np.abs(res1 - res2) < tolerance)
    print(are_equal)
    # index,res1_dif,res2_dif = find_differences_with_indices(res1, res2)
    # print()


if __name__ == "__main__":
    main()
