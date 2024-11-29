# Build linux kernel general dockerfile configuration file
FROM ubuntu:20.04

# Add apt sources
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i "s@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g" /etc/apt/sources.list \
&& sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

RUN mkdir -p /workspace/service_kws/
WORKDIR /workspace
COPY ./server.py /workspace/service_kws/
COPY ./cpp/sound_classification.so /workspace/service_kws/cpp/
COPY ./config.yaml /workspace/service_kws/
COPY ./requirements.txt .

# Add apt software
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    procps locales libncurses5 python3-dev python3-pip build-essential gcc g++ \
    && apt-get autoclean \
    && pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade \
    && python3 -m dfss --url=open@sophgo.com:tools/silk2/silk2.tools.logger-1.0.2-py3-none-any.whl \
    && pip3 install silk2.tools.logger-1.0.2-py3-none-any.whl \
    && pip3 install -r requirements.txt \
    && mkdir -p /workspace/service_kws/models/BM1688/ \
    && python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1688.tgz \
    && tar zxvf BM1688.tgz -C /workspace/service_kws/models/BM1688/ \
    && rm -f BM1688.tgz \
    && apt-get purge -y g++ gcc python3-pip build-essential \
    && apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && rm -rf silk2.tools.logger-1.0.2-py3-none-any.whl \
    && rm -rf requirements.txt \
    && rm -rf /root/.cache/* 

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
