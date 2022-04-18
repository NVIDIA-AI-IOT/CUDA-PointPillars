# Pointpillar Onnx Model Convertion
- This can be used to convert [pre-trained pointpillar model](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) into onnx file for TRT.

## CLI setup
```shell
$ docker pull scrin/dev-spconv
$ nvidia-docker run --rm -ti -v /home/$USER/:/workspace/ssh-docker --net=host scrin/dev-spconv
$ pip install pyyaml scikit-image onnx onnx-simplifier
$ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
$ python exporter.py --ckpt ./pointpillar_7728.pth
$ mv pointpillar.onnx ../model/ && mv params.h ../include/
```