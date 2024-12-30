#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# models
if [ ! -d "../models/" ]; 
then
    mkdir -p ../models/BM1688
    pushd ../models/BM1688
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1688_v2.tgz
    tar zxvf BM1688_v2.tgz
    rm -f BM1688_v2.tgz
    popd
    mkdir -p ../models/BM1684X
    pushd ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1684X_v2.tgz
    tar zxvf BM1684X_v2.tgz
    rm -f BM1684X_v2.tgz
    popd
    mkdir -p ../models/onnx
    pushd ../models/onnx
    python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/onnx_v2.tgz
    tar zxvf onnx_v2.tgz
    rm -f onnx_v2.tgz
    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

popd