## yolo11-xview (*Work In Progress*)

```
target=outputs/app/latest && docker cp yolo11-xview-yolo11-xview-1:/app/latest ./$target
```

<!--
target=out_1 && docker cp yolo11-xview-yolo11-xview-1:/app ./$target && docker cp yolo11-xview-yolo11-xview-1:/ultralytics/runs ./$target/runs


target=datasets/tgrs-hrrsd/autosplit_train.txt &&
docker cp yolo11-xview-yolo11-xview-1:/datasets/tgrs-hrrsd/autosplit_train.txt ./$target &&
target=datasets/tgrs-hrrsd/autosplit_val.txt &&
docker cp yolo11-xview-yolo11-xview-1:/datasets/tgrs-hrrsd/autosplit_val.txt ./$target


target=datasets/xview_chipped/images/autosplit_train.txt &&
docker cp yolo11-xview-yolo11-xview-1:/$target ./$target &&
target=datasets/xview_chipped/images/autosplit_val.txt &&
docker cp yolo11-xview-yolo11-xview-1:/$target ./$target


docker cp yolo11-xview-yolo11-xview-1:/datasets ./out_data

python src/process_wv.py "datasets/xView/train_images/" "datasets/xView/xView_train.geojson" && python src/process_wv.py "datasets/xView/val_images/" "datasets/xView/xView_train.geojson"
-->

### Datasets

- [xView](https://xviewdataset.org/)

### Training settings of known models

| Param         | [Yolov3](https://github.com/ultralytics/xview-yolov3/tree/main) |
| -             | -         |
| optim         | adam      |
| batch         | 16        |
| momentum      | 0.9       |
| decay         | 0.0005    |
| learning_rate | 0.001     |
| burn_in       | 1000      |
| image width   | 608       |
| image height  | 608       |

### Scheduled training runs

- [X] yolo11m pretrained on 20% of dataset
- [X] yolo11m from scratch on 20% of dataset
- [X] yolo11m with yolov3 training settings on 20% of dataset
- [X] yolo11m on 40%-60% of dataset
- [X] yolo11m on 80%-100% of dataset
