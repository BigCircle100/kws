import torch
from model import Cnn6

model = Cnn6(5, 8000)
model.load_state_dict(torch.load("model_weight/model_weights_5.pth"))
model.eval()
dummy_input = torch.randn(1,16000)

onnx_file_path = "kws.onnx"
torch.onnx.export(
    model,                # 要导出的模型
    dummy_input,         # 示例输入
    onnx_file_path,      # 输出文件名
    export_params=True,  # 是否导出参数
    opset_version=11,    # ONNX的操作集版本
    do_constant_folding=False,  # 是否进行常量折叠优化
    input_names=['input'],  # 输入名称
    output_names=['output'],  # 输出名称
    dynamic_axes={'input': {0: 'batch_size'},  # 动态批量大小
                  'output': {0: 'batch_size'}}  # 动态批量大小
)

