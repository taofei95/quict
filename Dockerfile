FROM ubuntu:18.04

WORKDIR /home/quict/

ENV LANG C.UTF-8

COPY . .

RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    apt update && \
    apt install git build-essential linux-headers-generic \
    python3.7 python3-pip python3-wheel python3-setuptools python3-numpy python3-scipy -y && \
    chmod +x build.sh install.sh && \
    ./build.sh && ./install.sh && \
    apt remove git build-essential linux-headers-generic python3-wheel python3-setuptools -y && \
    apt autoremove -y && \
    apt clean && \
    rm -rf build/ doc/