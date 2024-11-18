# 语音唤醒模块

## 目录

- [YOLOv5](#yolov5)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
    - [2.3 算法特性](#23-算法特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. YOLOv5 cpu opt](#7-yolov5-cpu-opt)
    - [7.1 NMS优化项](#71-nms优化项)
    - [7.2 精度测试](#72-精度测试)
    - [7.3 性能测试](#73-性能测试)
  - [8. FAQ](#8-faq)
  
## 1. 简介
语音唤醒模块基于[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn?tab=readme-ov-file)实现，目前支持的唤醒词包括：你好算能、清除缓存、清空缓存。

## 2 目录结构说明

```bash
.
├── cpp                           # cpp例程
├── readme.md           
└── scripts 
    ├── download.sh               # 例程模型与量化数据集
    └── gen_int8bmodel_mlir.sh    # onnx转bmodel脚本

```

## 3 模型训练
模型训练请参考[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn?tab=readme-ov-file)中提供的训练方法。您可以按照您的需求修改模型结构和参数量，并自行准备与您唤醒词有关的数据集。

**本例程中使用的模型为原仓库中的CNN10，对原模型的参数有一些修改，并将原仓库中提取特征的部分单独抽出来作为单独的类，因为后续使用tpu运行的是这个CNN部分。原始音频数据先经过以下类抽取特征后，将特征输入CNN模型获取分类结果。** 以下是一个示例，参数可以自行指定：
```c++
class ExtractFeature(nn.Module):
    def __init__(self, sample_rate, num_mel=40, window='hann', center=True, pad_mode='reflect',
                ref= 1.0, n_fft=1024, amin=1e-6, top_db = None, hop_length=256, fmin=0):
        super(ExtractFeature, self).__init__()
        n_mels = num_mel
        win_length = n_fft
        fmax = sample_rate // 2
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)
 
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft,
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True,is_log=True)
 
    def forward(self, inputs):
        x = self.spectrogram_extractor(inputs)
        logmel_spec = self.logmel_extractor(x)
        return logmel_spec
```

本例程中提供的训练模型，接受输入语音的长度为2s，采样率是8000。如果您使用了其他语音长度或采样率，或不同于上述特征提取的参数，您后续需要修改相关的推理代码（main.cpp中的sample_rate和time_len、SoundClassificationV2类中的audio_param_成员变量）以适配您自定义的训练参数。




## 4. 数据准备与模型编译

### 4.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[4.2 模型编译](#32-模型编译)进行模型转换。**


### 4.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。​同时，您需要准备用于量化的数据集。

**参照训练章节中的说明，您需要将已经把特征提取部分分离出来的CNN模型导出onnx，后续将该onnx导出bmodel。**

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。


- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数：

```shell
./scripts/gen_int8bmodel_mlir.sh 
```

​上述脚本会在`models/BM1688`文件夹下生成`nihaosuanneng.bmodel`，即转换好的INT8 BModel。

## 5. 例程测试

编译

运行

```bash
./soundclassification.soc ${input_file} ${bmodel_path}
```
请自行替换对应路径