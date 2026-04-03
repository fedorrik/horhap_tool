FROM ubuntu:22.04

MAINTAINER Fedor Ryabov, fryabov@ucsc.edu

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    bc \
    bedtools \
    && rm -rf /var/lib/apt/lists/*


RUN wget -qO- https://drive5.com/muscle/downloads3.8.31/muscle3.8.31_i86linux64.tar.gz | \
    tar -zxvf - && \ 
    mv muscle3.8.31_i86linux64 muscle3

WORKDIR /data
