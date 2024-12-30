#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
model_name=kws_v2
target=bm1688
target_dir=BM1688

if [ $# -gt 0 ]; then
    input=$1
    if [ "$input" == "bm1684x" ]; then
        target="bm1684x"
        target_dir="BM1684X"
    elif [ "$input" == "bm1688" ]; then
        target="bm1688"
        target_dir="BM1688"
    else
        echo "不支持的输入: $input"
        exit 1
    fi
fi

outdir=../models/${target_dir}

function gen_mlir()
{
    model_transform.py \
    --model_name ${model_name} \
    --model_def ../models/onnx/${model_name}.onnx \
    --input_shapes [[1,16000]] \
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