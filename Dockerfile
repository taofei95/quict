FROM silkeh/clang:15-bullseye AS builder
WORKDIR /tmp/quict

COPY . .

RUN echo "Building..." && \
    apt update && \
    apt install python3-pip git cmake -y && \
    python3 -m pip install pybind11 && \
    CC=clang && CXX=clang++ && bash ./build.sh

FROM python:3.9.15-slim-bullseye

WORKDIR /home/quict/

ENV LANG C.UTF-8

COPY --from=builder /tmp/quict/dist/ /tmp/quict/requirements.txt ./

RUN apt update && apt install vim libgomp1 -y && \
    python3 -m pip install -r ./requirements.txt && \
    python3 -m pip install ./QuICT*.whl && \
    python3 -m pip cache purge && \
    rm -rf *

ADD docker_run_script/scripts /home/quict/

CMD ["bash"]
