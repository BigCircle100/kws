#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# models
if [ ! -d "../models/" ]; 
then
    mkdir -p ../models/BM1688
    pushd ../models/BM1688
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1688.tgz
    tar zxvf BM1688.tgz
    rm -f BM1688.tgz
    popd
    mkdir -p ../models/onnx
    pushd ../models/onnx
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/onnx.tgz
    tar zxvf onnx.tgz
    rm -f onnx.tgz
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

if [ ! -d "../datasets/" ]; 
then 
    mkdir -p ../datasets/
    pushd ../datasets
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/processed_qingchu_huancun_common_voice_zh-CN_33497968.npy
    popd
else
    echo "Datasets folder exist! Remove it if you need to update."
fi
popd