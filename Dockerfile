FROM python:3.8-slim

WORKDIR /home/quict/

ENV LANG C.UTF-8

COPY . .

RUN apt update && \
    apt install git build-essential cmake python3-pip -y && \
    apt auto-remove -y &&\
    apt clean -y &&\
    bash ./dependency.sh &&\
    bash ./build.sh &&\
    bash ./install.sh

CMD ["bash"]