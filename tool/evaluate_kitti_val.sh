#!/bin/bash
python3 tool/eval/kitti_format.py
python3 tool/eval/evaluate.py evaluate --label_path=data/kitti/training/label_2/ --result_path=data/kitti/pred --label_split_file=tool/eval/val.txt --current_class=0,1,2 --coco=False