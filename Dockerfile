FROM alpine

WORKDIR /home/quict/

ENV LANG C.UTF-8

COPY . .

RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories && \
    sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories && \
    apk update && \
    apk add --no-cache bash linux-headers libtbb libtbb-dev gcc g++ make python3 py3-numpy py3-scipy py3-setuptools && \
    ./build.sh && \
    ./install.sh && \
    apk del gcc g++ make py3-setuptools libtbb-dev linux-headers && \
    rm -rf build build.sh install.sh .git .gitignore dependency.sh README.md doc QuICT setup.py Dockerfile


CMD ["bash"]