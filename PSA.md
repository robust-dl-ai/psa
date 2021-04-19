### Train a classification network to get CAMs.

```bash
python3 -m train_cls \
  --lr 0.1 \
  --batch_size 16 \
  --max_epoches 15 \
  --crop_size 448 \
  --network network.resnet38_cls \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --weights ../models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --wt_dec 5e-4 \
  --session_name ../outputs/psa/models/2021/resnet38_cls
```

### Inference for CAMs

```bash
python3 -m infer_cls \
  --infer_list voc12/train.txt \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --network network.resnet38_cls \
  --weights ../outputs/psa/models/2021/resnet38_cls.pth \
  --out_cam ../outputs/psa/data/interim/results/val/cam/ \
  --out_cam_pred ../outputs/psa/data/interim/results/val/cam_pred/ \
  --out_la_crf ../outputs/psa/data/interim/results/val/la_crf/ \
  --out_ha_crf ../outputs/psa/data/interim/results/val/ha_crf/
```

```bash
python3 -m infer_cls \
  --infer_list voc12/val.txt \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --network network.resnet38_cls \
  --weights ../outputs/psa/models/2021/resnet38_cls.pth \
  --out_cam ../outputs/psa/data/interim/results/val/cam/ \
  --out_cam_pred ../outputs/psa/data/interim/results/val/cam_pred/ \
  --out_la_crf ../outputs/psa/data/interim/results/val/la_crf/ \
  --out_ha_crf ../outputs/psa/data/interim/results/val/ha_crf/
```

### Train AffinityNet.

```bash
python3 -m train_aff \
  --lr 0.1 \
  --batch_size 8 \
  --max_epoches 8 \
  --crop_size 448 \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --network network.resnet38_aff \
  --weights ../models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --wt_dec 5e-4 \
  --la_crf_dir ../outputs/psa/data/interim/results/val/la_crf/ \
  --ha_crf_dir ../outputs/psa/data/interim/results/val/ha_crf/ \
  --session_name ../outputs/psa/models/2021/resnet38_aff
```


### Inference AffinityNet.

```bash
python3 -m infer_aff \
  --infer_list voc12/train.txt \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --network network.resnet38_aff \
  --weights ../outputs/psa/models/2021/resnet38_aff.pth \
  --cam_dir ../outputs/psa/data/interim/results/val/cam/ \
  --out_rw ../outputs/psa/data/interim/results/val/rw/
```

```bash
python3 -m infer_aff \
  --infer_list voc12/val.txt \
  --voc12_root ../data/raw/VOCdevkit/VOC2012 \
  --network network.resnet38_aff \
  --weights ../outputs/psa/models/2021/resnet38_aff.pth \
  --cam_dir ../outputs/psa/data/interim/results/val/cam/ \
  --out_rw ../outputs/psa/data/interim/results/val/rw/
```