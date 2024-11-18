#source tpu-mlir/envsetup.sh


chip="cv181x"

model_name="nihaosuanneng"
version_name="nihaosuanneng"

root="/workspace/models/audio_classification/nihaosuanneng"

build_dir="${root}/build_tmp"

img_dir="${root}/images"
img="${img_dir}/processed_qingchu_huancun_common_voice_zh-CN_33497968.npy"

mlir="${build_dir}/mlir/${version_name}_fp32.mlir"
table="${root}/calibration_table/${version_name}_cali_table"

int8="${build_dir}/int8/${version_name}_${chip}.cvimodel"

model_onnx="${root}/onnx/${model_name}.onnx"
in_npz="${build_dir}/${model_name}_in_f32.npz" #工具自动生成的，不要改路径
out_npz="${build_dir}/${version_name}_top_outputs.npz"


mkdir "${build_dir}"
mkdir "${build_dir}/mlir"
mkdir "${build_dir}/int8"
mkdir "${root}/calibration_table"

pushd $build_dir

if [ $1 = 1 -o $1 = "all" ] ; then
    model_transform.py \
    --model_name ${model_name} \
    --model_def ${model_onnx} \
    --test_input ${img} \
    --input_shapes [[1,1,126,40]] \
    --test_result ${out_npz}\
    --mlir ${mlir}
fi

if [ $1 = 2 -o $1 = "all" ] ; then

    run_calibration.py \
    ${mlir} \
    --dataset ${img_dir} \
    --input_num=149 \
    -o ${table}
fi

if [ $1 = 3 -o $1 = "all" ] ; then
    model_deploy.py \
    --mlir ${mlir} \
    --quantize INT8 \
    --quant_input \
    --calibration_table ${table} \
    --chip ${chip} \
    --tolerance 0.9,0.8,0.8 \
    --test_input ${in_npz}\
    --test_reference ${out_npz} \
    --model ${int8}
    # --debug
fi

popd
