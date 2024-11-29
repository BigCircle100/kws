cmake_minimum_required(VERSION 3.10)
project(sound_classification)

# 设置可执行文件输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE "Debug")

# 查找 libsophon
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 查找 pybind11
set(CMAKE_PREFIX_PATH "/home/linaro/.local/lib/python3.8/site-packages/pybind11/")
find_package(pybind11 REQUIRED)

include_directories(../3rdparty/)

# 收集源文件
aux_source_directory(. SRC_FILES)

# 添加共享库
pybind11_add_module(sound_classification ${SRC_FILES})

# 为目标添加 Eigen 的包含目录
target_include_directories(sound_classification PRIVATE ${EIGEN3_INCLUDE_DIRS})

# 链接库，使用关键字方式
target_link_libraries(sound_classification 
    PRIVATE  # 确保所有库使用相同的调用方式
    ${the_libbmlib.so} 
    ${the_libbmrt.so} 
    ${the_libbmcv.so} 
    pthread
)

# 设置共享库的输出路径
set_target_properties(sound_classification PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
