# Evaluation on Kitti

## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your data folder at `data/kitti`.
Note that in order to get the similar mAP compariable to OpenPCDet, we shall use pruned pointcloud in camera FOV.

The folder structure is as following:

```
data
    kitti
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

## Run evaluation kit

```
$ sh tool/evaluate_kitti_val.sh
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
```

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)
- [kitti_object_vis](https://github.com/kuixu/kitti_object_vis)
