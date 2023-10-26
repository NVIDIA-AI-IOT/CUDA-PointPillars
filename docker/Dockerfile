FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt install -y tzdata
RUN apt-get install -y python3.8 python3-pip git libgl1-mesa-glx libglib2.0-0

RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install pyyaml scikit-image onnx onnx-simplifier spconv==2.3.6 pillow==10.0.0 numba==0.58.0 opencv-python
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com