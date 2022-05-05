# Export Pointpillar Onnx Model 
- This can be used to convert [pre-trained pointpillar model](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) into onnx file for TRT.

## Requirements
All the codes are tested in the following environment:
* Ubuntu 18.04
* Python 3.8
* PyTorch 1.11.0
* CUDA 11.4

## CLI setup
Launch docker for spconv.
```shell
$ docker pull scrin/dev-spconv
$ nvidia-docker run --rm -ti -v /home/$USER/:/workspace/ssh-docker --net=host scrin/dev-spconv
```
Reinstall `pcdet v0.5` even if you have already installed previous version.
```shell
$ cd ../OpenPCDet
$ python setup.py develop
$ pip install pyyaml scikit-image onnx onnx-simplifier
$ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
$ python exporter.py --ckpt ./pointpillar_7728.pth
$ mv pointpillar.onnx ../model/ && mv params.h ../include/
```