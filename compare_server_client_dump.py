import os
import numpy as np

def compare_bin_files(dir1, dir2):
    for idx in range(1, 32):  # 从 1 到 31
        file1 = os.path.join(dir1, f"{idx}.bin")
        file2 = os.path.join(dir2, f"{idx}.bin")

        # 检查文件是否存在
        if not os.path.isfile(file1):
            print(f"File not found: {file1}")
            continue
        if not os.path.isfile(file2):
            print(f"File not found: {file2}")
            continue

        # 读取 bin 文件
        data1 = np.fromfile(file1, dtype=np.int16)
        data2 = np.fromfile(file2, dtype=np.int16)

        # 比较数据
        if np.array_equal(data1, data2):
            print(f"Files {idx}.bin are identical.")
        else:
            print(f"Files {idx}.bin are different.")

if __name__ == "__main__":
    dir1 = "client_test/client_cut/"
    dir2 = "server_dump/"
    
    compare_bin_files(dir1, dir2)
