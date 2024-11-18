#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

target=bm1688
target_dir=BM1688
model_name=nihaosuanneng
img_dir=../datasets/images
img=../datasets/images/processed_qingchu_huancun_common_voice_zh-CN_33497968.npy

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

function gen_cali_table(){
    run_calibration.py \
    ${model_name}_fp32.mlir \
    --dataset ${img_dir} \
    --input_num=149 \
    -o ${model_name}_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
    --mlir ${model_name}_fp32.mlir \
    --quantize INT8 \
    --quant_input \
    --calibration_table ${model_name}_cali_table \
    --chip ${target} \
    --tolerance 0.9,0.8,0.8 \
    --test_input ${model_name}_in_f32.npz \
    --test_reference ${model_name}_top.npz \
    --model ${model_name}.bmodel
    # --debug

    mv ${model_name}.bmodel $outdir/
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

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${model_name}_fp32.mlir \
        --quantize F32 \
        --chip ${target} \
        --model ${model_name}_fp32.bmodel
    mv ${model_name}_fp32.bmodel $outdir/
}

pushd ${model_dir}
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_mlir
# gen_cali_table
# gen_int8bmodel
# gen_fp16bmodel
gen_fp32bmodel

popd