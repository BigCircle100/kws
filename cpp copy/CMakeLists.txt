cmake_minimum_required(VERSION 3.10)
project(sound_classification)

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE "Debug")

# 查找 libsophon
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 收集源文件
aux_source_directory(. SRC_FILES)

# 添加可执行文件
add_executable(sound_classification.soc ${SRC_FILES})

# 为目标添加 Eigen 的包含目录
target_include_directories(sound_classification.soc PRIVATE ${EIGEN3_INCLUDE_DIRS})

# 链接库
target_link_libraries(sound_classification.soc 
    ${the_libbmlib.so} 
    ${the_libbmrt.so} 
    ${the_libbmcv.so} 
    pthread  # 不需要加前缀 -l
)
