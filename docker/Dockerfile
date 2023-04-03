## Basic Docker Enviroment for correct OpenPCDet for CUDA-PointPillars
##
## Prerequisites:
##   - docker
##   - nvidia driver
##   - nvidia container toolkit
##
## Build with:
##   docker build -t open_pcdet:latest .
##
## Save with:
##   docker save  open_pcdet:latest  | gzip > open_pcdet.tar.gz
##
## Run with:
##   docker run --network host --gpus all --name PointPillars --rm -it  open_pcdet:latest
## Debug with:
##   docker run --network host --gpus all --name PointPillars --entrypoint /bin/bash -it  open_pcdet:latest

FROM nvcr.io/nvidia/pytorch:20.12-py3

## Meta information
LABEL cuda.version="11.1" maintainers="Robert Belshaw <rbelshaw@sagarobotics.com>"

WORKDIR /
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN pip install --upgrade pip wheel setuptools
RUN pip install spconv-cu113
RUN git clone --branch v0.5.2 https://github.com/open-mmlab/OpenPCDet.git
WORKDIR /OpenPCDet
RUN python setup.py develop
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6  -y
RUN DEBIAN_FRONTEND=noninteractive pip install onnxsim nvidia-pyindex  
RUN DEBIAN_FRONTEND=noninteractive pip install onnx-graphsurgeon
RUN DEBIAN_FRONTEND=noninteractive pip install kornia==0.4.1

# Docker clean-up
RUN rm -rf /var/lib/apt/lists/*

## Create entry point for image
WORKDIR /
COPY ./entrypoint.sh .
ENTRYPOINT ["/bin/bash"]
CMD ["/bin/bash", "-c"]