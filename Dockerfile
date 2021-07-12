FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	build-essential \
	libopencv-dev \
	python3-pip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt