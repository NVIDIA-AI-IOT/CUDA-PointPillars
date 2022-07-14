# KITTI Evaluation Kit

## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your data folder at `kitti/object`.
Note that in order to get the similar mAP compariable to OpenPCDet, we shall use pruned pointcloud in camera FOV.

The folder structure is as following:

```
kitti
    object
        pcdet
            000000.txt
        pred
            000000.txt
        pred_velo
            000000.txt
        testing
            calib
               000000.txt
            image_2
               000000.png
            label_2
               000000.txt
            velodyne
               000000.bin
            pred
               000000.txt
        training
            calib
               000000.txt
            image_2
               000000.png
            label_2
               000000.txt
            velodyne
               000000.bin
            pred
               000000.txt
```

## Setup Conda Environment

### Install spconv-1.0

- Start from a new conda enviornment:

```
(base)$ conda create -n kitti_eval python=3.6
(base)$ conda activate kitti_eval
(kitti_eval)$ conda install pytorch==1.0 cudatoolkit=10.0 -c pytorch
(kitti_eval)$ conda install cudnn boost
```

- Fetch spconv-1.0 code

```
(kitti_eval)$ git clone https://github.com/traveller59/spconv spconv_8da6f96 --recursive
(kitti_eval)$ cd spconv_8da6f96/
(kitti_eval)$ git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634
(kitti_eval)$ git submodule update --init --recursive
```

- Build wheel and install spconv-1.0

```
(kitti_eval)$ python setup.py bdist_wheel
(kitti_eval)$ cd dist/ && pip install *.whl
```

### Install second-1.5.1

- Fetch second-1.5.1 code and install dependency

```
(kitti_eval)$ git clone https://github.com/traveller59/second.pytorch.git
(kitti_eval)$ cd second.pytorch/
(kitti_eval)$ git checkout v1.5.1
(kitti_eval)$ pip install scikit-image scipy numba pillow matplotlib
(kitti_eval)$ pip install fire tensorboardX protobuf opencv-python
(kitti_eval)$ export PYTHONPATH=/path_to_second.pytorch:$PYTHONPATH
```

### Run evaluation kit

- Format prediction and pcdet data into kitti format

```
(kitti_eval)$ cd /path_to_eval
(kitti_eval)$ python kitti_format.py
```

- Run evaluation kit on prediction and pcdet outputs

```
(kitti_eval)$ python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pred --label_split_file=./val.txt --current_class=0,1,2 --coco=False
Car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.78, 89.80, 88.74
bev  AP:89.48, 86.99, 83.86
3d   AP:86.28, 77.08, 73.87
aos  AP:90.77, 89.60, 88.42
Car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.78, 89.80, 88.74
bev  AP:90.78, 90.15, 89.42
3d   AP:90.78, 90.02, 89.19
aos  AP:90.77, 89.60, 88.42
Pedestrian AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:65.66, 61.90, 58.44
bev  AP:61.02, 56.29, 52.60
3d   AP:56.49, 51.68, 47.55
aos  AP:47.03, 44.53, 41.78
Pedestrian AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:65.66, 61.90, 58.44
bev  AP:72.18, 69.56, 66.12
3d   AP:72.11, 68.94, 65.84
aos  AP:47.03, 44.53, 41.78
Cyclist AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:85.04, 72.64, 68.80
bev  AP:82.38, 65.88, 61.53
3d   AP:80.35, 62.56, 59.19
aos  AP:84.50, 70.79, 66.93
Cyclist AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:85.04, 72.64, 68.80
bev  AP:86.15, 70.29, 66.71
3d   AP:86.15, 70.29, 66.71
aos  AP:84.50, 70.79, 66.93

(kitti_eval)$ python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pcdet/ --label_split_file=./val.txt --current_class=0,1,2 --coco=False
Car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.77, 89.77, 88.75
bev  AP:89.52, 87.06, 84.10
3d   AP:85.97, 77.10, 74.41
aos  AP:90.76, 89.57, 88.42
Car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.77, 89.77, 88.75
bev  AP:90.78, 90.15, 89.42
3d   AP:90.78, 90.03, 89.19
aos  AP:90.76, 89.57, 88.42
Pedestrian AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:66.17, 62.15, 59.16
bev  AP:61.32, 56.09, 52.52
3d   AP:56.57, 51.93, 47.42
aos  AP:48.22, 45.30, 42.78
Pedestrian AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:66.17, 62.15, 59.16
bev  AP:72.09, 69.09, 66.09
3d   AP:72.01, 68.86, 65.06
aos  AP:48.22, 45.30, 42.78
Cyclist AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:85.04, 72.50, 68.54
bev  AP:82.01, 66.16, 62.33
3d   AP:79.74, 62.43, 59.37
aos  AP:84.49, 70.66, 66.65
Cyclist AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:85.04, 72.50, 68.54
bev  AP:86.25, 70.30, 66.33
3d   AP:86.25, 70.30, 66.33
aos  AP:84.49, 70.66, 66.65

```

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)
- [kitti_object_vis](https://github.com/kuixu/kitti_object_vis)
