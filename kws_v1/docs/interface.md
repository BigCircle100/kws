# 服务层接口和请求体

- 启动kws服务
    - 设置日志级别：debug, info, warning, error

- 获取状态
    - 发送post请求：http://ip:port/status
    - 返回的消息格式
        ```js
        {
            "messageType": "get_status"
            "status": 1
        }
        ```
        - 参数说明
            - status：1为初始化完毕，2为推理中，3为推理完成


- 获取模型参数
    - 发送post请求：http://ip:port/get_param
    - 返回的消息格式
        ```js
        {
            "messageType": "get_param"
            "threshold": 0.5
        }
        ```
        - 参数说明
            - threshold：模型topk求出的最大值如果低于这个值，代表不可信，则默认分类结果是背景而不是唤醒词

- 设置模型参数
    - 发送post请求：http://ip:port/set_param
    - 请求格式
        ```js
        {
            "messageType": "set_param"
            "threshold": 0.5
        }
        ```
    - 设置完成返回response 200
        - 参数说明
            - threshold：同获取模型参数

- 发送语音推理
    - 建立websocket：ws://ip:port/inference
    - 首次发送请求格式
        ```js
        {
            "meassgeType": "inference",
            "param": {
                "format": "INT16",
                "rate": 44100,
                "time_per_buffer": 200,
                "channels": 1
            }
        }
        ```
        - 参数说明：
            - format：发送语音的数据格式，目前只支持INT16
            - rate：发送语音的采样率
            - time_per_buffer：发送的每段语音的长度，同时也是每次发送的时间间隔，单位是ms
            - channels：语音的通道数，目前只支持单通道
        - 错误返回格式：
            ```js
            {
                "messageType": "inference",
                "code": 1
            }
            ```
            - code为错误码，错误码说明：
                - 1：首次发送的请求体中没有messageType字段
                - 2：messageType字段不是"inference"
                - 3：param字段中有错误
            
    - 后续发送请求格式
        - 二进制语音数据

    - 返回推理结果格式
        ```js
        {
            "result":1,
            "prob_avg": 0.65,
            "prob_max": 0.65
        }
        ```
        - 参数说明：
            - result： 1为你好算能，2为清除缓存，3为清空缓存，如果没有关键词就不返回
            - prob_avg: 是result分类的平均概率（多次识别到是这个类以后才会返回，对这些次的概率取平均）
            - prob_max: 是result分类的最大概率（同上，取最大）

- 心跳
    - 发送get请求：http://ip:port/health
    - 存活返回response 200