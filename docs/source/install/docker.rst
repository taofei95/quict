
Docker
-----------------------------------------------------------------------

In case that you have a favor over Docker
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Though releasing images on Docker Hub is in our agenda, currently docker users might need to build docker image from sources. With the docker file we provide in the repository, one can easily build a docker image with only a little performance loss.

  docker build -t quict .

docker build file
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
