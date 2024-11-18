#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.signal import resample
import json
import argparse
import yaml
import uvicorn
import sys
import os
import numpy as np
import time
import math
from threading import Lock
from queue import Queue
from typing import Dict, List

import SILK2.Tools.logger as Logger

from scipy.io.wavfile import write

lib_path = os.path.abspath('./cpp')
sys.path.append(lib_path)

from cpp.sound_classification import SoundClassificationV2


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

input_span = 2
target_rate = 8000
# 推理请求间隔
client: SoundClassificationV2

active_websockets: List[WebSocket] = []


class KWSStatusResponse(BaseModel):
    messageType: str
    status: int


class ModelParam(BaseModel):
    messageType: str
    threshold: float



@app.post("/status", response_model=KWSStatusResponse)
async def send_status():
    global client
    cur_status = client.get_status()

    msg = KWSStatusResponse(messageType="get_status", status=cur_status)
    return msg


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/get_param", response_model=ModelParam)
async def get_param():
    global client
    threshold = client.get_threshold()
    return ModelParam(
        messageType="get_param",
        threshold=threshold
    )

@app.post("/set_param")
async def set_param(request: ModelParam):
    global client
    client.set_threshold(request.threshold)

    return Response(status_code=200)


@app.websocket("/inference")
async def model_inference(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    buffer_count = 0
    time_per_buffer = 200   # 单位ms
    origin_rate = 44100
    count = 0

    res_prob = {1: [],       # nihaosuannneg
                2: [],       # qingchuhuancun
                3: []}       # qingkonghuancun
    ret_count = 2           # 某类的重复次数大于等于这个数，会给中台返回结果
    infer_time = 300        # 预设推理时长300ms，决定接受几次二进制包推理一次，比如每个包内容是200ms，那么发两次再推1次

    global client
    global logger

    total_data = np.array([], dtype=np.int16)

    try: 
        while True:
            data = await websocket.receive()
            # 语音数据开头是
            if "text" in data.keys():
                message = json.loads(data["text"])  # 将接收到的 JSON 字符串解析为字典
                
                # message_type = message.get("messageType")
                param = message.get("param", {})
                format_type = param.get("format")
                origin_rate = param.get("rate")
                time_per_buffer = param.get("time_per_buffer")
                channels = int(param.get("channels"))

                if format_type != "INT16":
                    raise NotImplementedError("only support int16 data type for now")
            
                if channels != 1:
                    raise NotImplementedError("only support single channel for now")

                total_data = np.array([], dtype=np.int16)
                for key in res_prob:
                    res_prob[key].clear()
            elif "bytes" in data.keys():
                start_np = time.time()
                byte_data = data["bytes"]

                new_data_origin = np.frombuffer(byte_data, dtype=np.int16)

                target_num_samples = int(len(new_data_origin)*target_rate/origin_rate)
                new_data = resample(new_data_origin, target_num_samples).astype(np.int16)


                total_data = np.concatenate((total_data, new_data))
                if (len(total_data) >= 16000):
                    total_data = total_data[len(total_data)- 16000:]
                else:
                    continue
                input_data = np.copy(np.ascontiguousarray(total_data))


                end_np = time.time()
                logger.debug(f'create input_data time: {end_np-start_np}')

                buffer_count += 1

                # if True:
                if buffer_count*time_per_buffer >= infer_time:
                    buffer_count = 0
                    start_infer = time.time()

                    count += 1
                    logger.debug(f'infer count: {count}')
                    # input_data.tofile(f"server_dump/{count}.bin")
                    res, prob = client.inference(input_data)
                    logger.debug(f'res: {res}, prob: {prob}')
                    end_infer = time.time()
                    logger.debug(f'inference time: {end_infer-start_infer}')

                    # 连续几次识别中，如果有一次分类成背景，就清空计数
                    # 背景类的召回率比较低，容易将应该分类成背景的非关键词，错误的分成了关键词类
                    # 精确率很高，如果分类成背景，一定更可信，就将前面的计数清除
                    if res == 0:
                        for key in res_prob:
                            res_prob[key].clear()
                    else:
                        res_prob[res].append(prob)
                        for key, value in res_prob.items():
                            if len(value) >= ret_count:
                                infer_result = json.dumps({"result": key, "prob_avg": sum(value)/len(value), "prob_max": max(value)})
                                logger.info(f"successfully classified: [result: {key}, prob_avg: {sum(value)/len(value)}, prob_max: {max(value)}]")
                                await websocket.send_text(infer_result)
                                return 
            
    except Exception as e:
        logger.error(f"{Logger.file_lineno()} {str(e)}",exc_info=True)
    finally:
        active_websockets.remove(websocket)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    curr_path = os.path.dirname(os.path.abspath(__file__))
    default_yaml = os.path.join(curr_path,"config.yaml")
    parser.add_argument('--config', type=str, default=default_yaml, help='path of config file')
    parser.add_argument('--loglevel', type=str, default="info", help="set log level")
    parser.add_argument('--ip', type=str, default="127.0.0.1", help="set log level")
    parser.add_argument('--port', type=int, default=18080, help="set log level")

    args = parser.parse_args()
    return args

def is_running(pid_file):
    """检查是否有其他实例在运行"""
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
            try:
                # 检查进程是否存在
                os.kill(pid, 0)
                return True  # 进程仍在运行
            except OSError:
                return False  # 进程不存在
    return False


if __name__ == "__main__":
    pid_file_path = '/tmp/kws_service.pid'
    if is_running(pid_file_path):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%d-%H:%M:%S", current_time)
        print(formatted_time,": 语音唤醒 服务已经在运行。")
        sys.exit(1)

    # 写入当前进程的 PID
    with open(pid_file_path, 'w') as f:
        f.write(str(os.getpid()))
    
    try:    
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%d-%H:%M:%S", current_time)
        print("语音唤醒 服务启动:",formatted_time)   
        # 写入当前进程的 PID
        curr_path = os.path.dirname(os.path.abspath(__file__))
        
        args = argsparser()
        ip = args.ip
        port = args.port
        loglevel = args.loglevel.upper()
        # logger = Logger.Log("service_kws", loglevel)
        logger = Logger.Log("service_kws", "DEBUG")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        models_config = config["models"]        

        init_config = {}
        for model in models_config:
            for key, value in model.items():
                if key.find('path') >= 0:
                    if value[:2] == './':
                        value = os.path.join(curr_path, value[2:])
                    elif value[0] != '/':
                        value = os.path.join(curr_path, value)
                model[key] = value

            name = model["name"]
            if "nihaosuanneng" in name:
                client = SoundClassificationV2(model["dev_id"],model["bmodel_path"],model["threshold"])
                client.set_logger_level(loglevel)
            else:
                print(f"The model {name} is not yet adapted")          

        uvicorn.run(app, host=ip, port=port, workers=1)

    except KeyboardInterrupt:
        print("程序被中断。")
    finally:
        # 删除 PID 文件
        os.remove(pid_file_path)


