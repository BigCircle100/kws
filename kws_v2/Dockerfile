ARG PLATFORM="bm1688"

# Build linux kernel general dockerfile configuration file
FROM ubuntu:20.04

# Add apt sources
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i "s@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g" /etc/apt/sources.list \
&& sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

RUN mkdir -p /workspace/service_kws/
WORKDIR /workspace
COPY ./server.py /workspace/service_kws/
COPY ./model.py /workspace/service_kws/
COPY ./requirements.txt .

COPY config_bm1684x.yaml /workspace/
COPY config_bm1688.yaml /workspace/
COPY ./start_run.sh  /workspace/service_kws/

# Add library path
ENV PYTHONPATH "$PYTHONPATH:/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-opencv-latest/opencv-python/"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-opencv-latest/lib:/opt/sophon/sophon-ffmpeg-latest/lib:/opt/sophon/sophon-soc-libisp_1.0.0/lib"

ARG PLATFORM

RUN if [ "$PLATFORM" = "bm1688" ]; then \
        cp config_bm1688.yaml /workspace/service_kws/config.yaml; \
    elif [ "$PLATFORM" = "bm1684x" ]; then \
        cp config_bm1684x.yaml /workspace/service_kws/config.yaml; \
    else \
        echo "Error: Unsupported platform ${PLATFORM}." && exit 1; \
    fi

RUN rm config_*

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
    && if [ "${PLATFORM}" = "bm1688" ]; then \
        mkdir -p /workspace/service_kws/models/BM1688/ \
        && python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1688_v2.tgz \
        && tar zxvf BM1688_v2.tgz -C /workspace/service_kws/models/BM1688/ \
        && rm -f BM1688_v2.tgz \
        && python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/sophon_arm-3.8.0-py3-none-any.whl \
        && pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall \
        && rm -f sophon_arm-3.8.0-py3-none-any.whl; \
    elif [ "${PLATFORM}" = "bm1684x" ]; then \
        mkdir -p /workspace/service_kws/models/BM1684X/ \
        && python3 -m dfss --url=open@sophgo.com:SILK/level-3/service_kws/BM1684X_v2.tgz \
        && tar zxvf BM1684X_v2.tgz -C /workspace/service_kws/models/BM1684X/ \
        && rm -f BM1684X_v2.tgz \
        && python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/84x_soc_sail/sophon-3.8.0-py3-none-any.whl \
        && pip3 install sophon-3.8.0-py3-none-any.whl --force-reinstall \
        && rm -f sophon-3.8.0-py3-none-any.whl; \
    else \
        echo "error: get ${PLATFORM}, only support platform: bm1684X|bm1688" && exit 1; \
    fi \
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

WORKDIR /workspace/service_kws
ENTRYPOINT ["/bin/bash", "start_run.sh"]