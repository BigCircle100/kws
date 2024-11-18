板端ip: 172.25.3.22, 用户名：root 密码：cvitek
执行：

export LD_LIBRARY_PATH=/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ive_sdk/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_tpu_sdk/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib/3rd:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib:/mnt/data/yzx/sdk_new_xgit/middleware/v2/lib:/mnt/data/yzx/sdk_new_xgit/middleware/v2/lib/3rd:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/sample/3rd/lib:/mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/sample/3rd/rtsp/lib
cd /mnt/data/yzx/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/bin
/mnt/data/yzx/sdk_new_xgit/ramdisk/rootfs/public/gdbserver/riscv_musl/usr/bin/gdbserver localhost:60000 ./test_audio_cls /mnt/data/yzx/infer/sound/models/nihaosuanneng_cv180x.cvimodel /mnt/data/yzx/infer/sound/data/nihaosuanneng/1/processed_hello_sn_common_voice_zh-CN_33480510.bin 8000 2 0.5


PC端ip：10.80.39.3 用户名和密码均为 algo.public
执行：
cd /home/zhenxing.ye/workspace/nfsuser/sdk_new_xgit/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/bin
/home/zhenxing.ye/workspace/nfsuser/sdk_new_xgit/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gdb ./test_audio_cls

target remote 172.25.3.22:60000



板子挂载路径/mnt/data/public/xin.chen -> /data/algo.public/nfsuser/xin.chen
现在板子上操作


export LD_LIBRARY_PATH=/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ive_sdk/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_tpu_sdk/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib/3rd:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/usr/lib:/data/algo.public/nfsuser/xin.chen/sophpi/cvi_mpi/lib:/data/algo.public/nfsuser/xin.chen/sophpi/cvi_mpi/lib/3rd:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/rootfs/mnt/system/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/sample/3rd/lib:/mnt/data/public/xin.chen/sophpi/install/soc_cv1801c_wevb_0009a_spinor/tpu_musl_riscv64/cvitek_ai_sdk/sample/3rd/rtsp/lib



-------
export LD_LIBRARY_PATH=/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/tpu_musl_riscv64/cvitek_ive_sdk/lib:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/rootfs/mnt/system/usr/lib:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/tpu_musl_riscv64/cvitek_ai_sdk/lib:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/tpu_musl_riscv64/cvitek_tpu_sdk/lib:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/rootfs/mnt/system/usr/lib/3rd:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/rootfs/mnt/system/usr/lib:/mnt/data/public/sophpi/cvi_mpi/lib:/mnt/data/public/sophpi/cvi_mpi/lib/3rd:/mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/tpu_musl_riscv64/cvitek_ai_sdk/sample/3rd/rtsp/lib

cd /mnt/data/public/sophpi/install/soc_cv1811h_wevb_0007a_spinand/tpu_musl_riscv64/cvitek_ai_sdk/bin
