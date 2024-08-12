# Efficient Hybrid Feature Interaction Network for Stereo Image Super-Resolution

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
cd code
pip install -r requirements.txt
python setup.py develop
```
## Datasets
- EHFSSR/EHFSSR_S

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  Flickr1024 + Middlebury | KITTI2012 + KITTI2015 + Middlebury + Flickr1024  |

Refer to the related references in the manuscript for the complete data. 

## Implementation of EHFSSR/EHFSSR-S
### Train

```shell
#generate .h5 file
python scripts/generateh5.py
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EHFSSR/EHFSSR_x2_s1.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EHFSSR/EHFSSR_x2_s2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EHFSSR/EHFSSR_x4_s1.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EHFSSR/EHFSSR_x4_s2.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/EHFSSR/EHFSSR_x2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/EHFSSR/EHFSSR_x4.yml --launcher pytorch
```
