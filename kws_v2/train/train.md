# 语音唤醒模块模型训练

## 1. 模型详情
当前模块基于pytorch，使用的模型基于[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn?tab=readme-ov-file)做了一些修改，模型定义可以参考[model.py](./model.py)中的Cnn6。当前有5种类别：
```
0：背景/非关键词
1：你好算能
2：清除缓存
3：清空缓存
4：hello silk
```
当前模型输入是采样率8000的2秒语音数据。您可以根据需求自行更换或调整模型，保证模型输出类别也是从0开始的数字即可。

## 2. 数据集和环境准备
请自行准备唤醒词和非唤醒词类的数据集，对于当前例程使用的模型，建议每类数据2000+。当前例程的训练数据满足每条数据采样率8000，时长2秒。数据集目录结构示例如下：
```
dataset
├── train
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   └── 4
└── val
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    └── 4
```
数据集读取细节请参考[dataset.py](./dataset.py)，其中__getitem__方法添加了一些预处理，您可以根据自己模型的实际情况修改或删除。

当前模型的训练环境请参考[requirements.txt](./requirements.txt)。


## 3. 训练
训练脚本请参考[train.py](./train.py)。可修改其中的模型或训练参数，例如：
```python
# 数据集目录
root_dir = "dataset/"
# 权重保存目录
model_save = f"model_weight/"

# 模型训练相关参数
sample_rate = 8000
learning_rate = 0.001
num_epochs = 5
batch_size = 32
num_classes = 5     # 根据您唤醒词/非唤醒词的类别总数自行修改
start_epoch = 0
```

## 4. 导出onnx
模型导出onnx请参考[export.py](./export.py)，请自行修改需要导出的模型权重路径，以及onnx文件名。


