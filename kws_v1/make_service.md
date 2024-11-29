通过dockerfile构建镜像，并启动容器直接运行服务的方法如下：（这里以kws服务为例，运行目录是service_kws）

# 1. 准备的文件和目录关系

在构建docker镜像之前，先保证silk2.tools.logger-1.0.2-py3-none-any.whl和requirements.txt已经在你的目录下了：
（仅列出了相关文件）

```bash
service_kws
├── ...
├── Dockerfile
├── requirements.txt
└── silk2.tools.logger-1.0.2-py3-none-any.whl

```

# 2. dockerfile内容
如果有其他需要安装的内容，可以在最后的部分继续添加。如果安装文件在本地，请先执行copy命令将文件考到镜像中才可以正确安装，如下面的wheel和requirements.txt。还需要将service的所有代码和模型拷贝到镜像的/workspace下，请自行修改/workspace下的目录名

```bash
# Build linux kernel general dockerfile configuration file
FROM ubuntu:20.04

# Add apt sources
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i "s@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g" /etc/apt/sources.list \
&& sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

# Add apt software
RUN export DEBIAN_FRONTEND=noninteractive \
&& apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y --no-install-recommends \
procps locales libncurses5 python3-dev python3-pip build-essential gcc g++\
&& apt-get autoclean

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Add library path
ENV PYTHONPATH "$PYTHONPATH:/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-opencv-latest/opencv-python/"

# Install silk2.tools.logger and other dependencies
WORKDIR /app
COPY silk2.tools.logger-1.0.2-py3-none-any.whl .
COPY requirements.txt .
RUN export DEBIAN_FRONTEND=noninteractive \
&& pip3 install silk2.tools.logger-1.0.2-py3-none-any.whl \
&& pip3 install -r requirements.txt

# Copy server code and model to /workspace, change the name to your own service
RUN mkdir /workspace
COPY . /workspace/service_kws/

```

# 3. 构建镜像
在dockerfile所在目录执行，请自行修改命令中的镜像名：
```bash
docker build -t service_kws:1.1.0 .
```
上述命令构建的镜像名是service_kws:1.1.0，可以通过docker image ls查看到。在每次修改代码生成新的镜像时，请更新版本号。

# 4. 启动容器并同时运行服务
当前server的服务端口是18080。下面命令中，容器除了挂载了运行环境依赖的路径以外，还映射了服务端口，以及log目录（每个服务需要在/data下有自己的目录保存日志），并直接切换到服务所在目录运行服务。请自行修改相关参数和执行的命令。

```bash
sudo docker run \
-td \
--privileged=true \
-v /opt/sophon:/opt/sophon \
-v /dev:/dev \
-v /etc/profile.d:/etc/profile.d \
-v /etc/profile:/etc/profile \
-v /etc/ld.so.conf.d:/etc/ld.so.conf.d \
-v /data/kws_log:/data/kws_log \
-p 18080:18080 \
--name service_kws service_kws:1.1.0 /bin/bash -c "cd /workspace/service_kws && python3 server.py > /data/kws_log/kws_log_$(date +%Y-%m-%d).txt 2>&1"
```
正常执行完，这个容器会持续运行，直到server停止。如果容器刚运行就退出了，说明执行命令运行出错了。可以先以交互式进入容器，手动运行服务相关命令，验证命令没问题后再删除当前容器，用上述命令重新启动容器与服务。
