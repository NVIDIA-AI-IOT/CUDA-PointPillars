The tool can convert network model of Pointpillars into the onnx file which can be used for TRT.

Steps:
1. $ docker pull scrin/dev-spconv

2. $ nvidia-docker run --rm -ti -v /home/$USER/:/workspace/ssh-docker --net=host  scrin/dev-spconv

3. $ pip install pyyaml
$ pip install scikit-image
$ pip install onnx onnx-simplifier 
$ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

4. $ python exporter.py --ckpt {*.pth}
$ python exporter.py --ckpt ./pointpillar_7728.pth

You can get a onnx files:"pointpillar.onnx" in current folder.

