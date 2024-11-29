#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

target=bm1688
target_dir=BM1688
model_name=suan_neng_v3
img=../datasets/processed_qingchu_huancun_common_voice_zh-CN_33497968.npy

outdir=../models/${target_dir}

function gen_mlir()
{
    model_transform.py \
    --model_name ${model_name} \
    --model_def ../models/onnx/${model_name}.onnx \
    --test_input ${img} \
    --input_shapes [[1,1,126,40]] \
    --test_result ${model_name}_top.npz\
    --mlir ${model_name}_fp32.mlir
}


function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_fp32.mlir \
        --quantize F16 \
        --chip ${target} \
        --model ${model_name}_fp16.bmodel
    mv ${model_name}_fp16.bmodel $outdir/
}

pushd ${model_dir}
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
gen_fp16bmodel


popd