FROM docker.io/library/ubuntu:latest
RUN apt-get update -q
RUN apt-get upgrade -yqq
RUN apt-get install -y python3.10 python3-pip
RUN pip3 install torch
WORKDIR /root