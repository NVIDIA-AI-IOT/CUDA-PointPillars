The tool can convert network model of Pointpillars into the onnx file which can be used for TRT.

Steps:
1. $ docker pull scrin/dev-spconv
OR
$ docker pull scrin/dev-spconv:f22dd9aee04e2fe8a9fe35866e52620d8d8b3779

2. $ nvidia-docker run --rm -ti -v /home/$USER/:/workspace/ssh-docker --net=host  scrin/dev-spconv
OR
$ sudo docker run --rm --gpus all -ti -v /home/$USER/:/workspace/ssh-docker --net=host  scrin/dev-spconv:f22dd9aee04e2fe8a9fe35866e52620d8d8b3779

3. $ pip install pyyaml
$ pip install scikit-image
$ pip install onnx onnx-simplifier 
$ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

4. $ python exporter.py --ckpt {*.pth}
$ python exporter.py --ckpt ./pointpillar_7728.pth

You can get a onnx files:"pointpillar.onnx" and "params.h" in current folder.
Please:
copy "pointpillar.onnx" into ${project/path}/model
copy "params.h" into ${project/path}/include
