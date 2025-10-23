# NeurIPS25-D&B-Submission

This is the detailed readme for our code submission on paper "Benchmarking Semi-Supervised Semantic Segmentation Models for Post-Disaster Scenes Understanding".

## Dataset

You can access the fully-supervised Floodnet dataset at [this](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=3&dl=0) link. Fully-supervised RescueNet dataset can be found at [this](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1) link.

We use the semi-supervised version from [this](https://github.com/BinaLab/RescueNet-Challenge2023?tab=readme-ov-file) link (RescueNet) and [this](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) link (FloodNet).


## ReCo

In order to run ReCo, you may need [Accelerate](https://huggingface.co/docs/accelerate/en/index) besides those standard PyTorch and computer vision packages.

You need to organized the dataset like following:
```
├── ./dataset
    ├── floodnet
        ├── trainset
          ├── train-org-img
          ├── train-label-img
        ├── validationset
          ├── val-org-img
          ├── val-label-img
        ├── train_labeled.txt
        ├── train_unlabeled.txt
        ├── val.txt
    ├── rescuenet
        ├── trainset_all
            ├── train-org-img
            ├── train-label-img
        ├── validationset
            ├── val-org-img
            ├── val-label-img
        ├── train_labeled.txt
        ├── train_unlabeled.txt
        ├── val.txt
```
You may also need to double check the file path in the "build_data.py"

To train ReCo on RescueNet and FloodNet:
```
nohup accelerate launch train_semisup_acc.py --dataset rescuenet --apply_aug classmix --apply_reco

nohup accelerate launch train_semisup_acc.py --dataset floodnet --apply_aug classmix --apply_reco > floodnet_reco.log 2> err.log & 
```

To evaluate trained ReCo on RescueNet and FloodNet:
```
python eval.py
```

## S4MC

You need to create a new conda environment using the provided "environment.yml".
If you get any error with PyTorch version, you may need to run
```
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
conda env update --file environment.yml --prune
```
You may also need the headless opencv installation.

You need to organized the dataset like following:
```
├── ./data
    ├── FloodNet
        ├── Train
            ├── label-img
            ├── org-img
        ├── Validation
            ├── label-img
            ├── org-img
        ├── floodnet-labeled.txt
        ├── floodnet-unlabeled.txt
        ├── floodnet-val.txt
    ├── RescueNet
        ├── Train
            ├── label-img
            ├── org-img
        ├── Validation
            ├── label-img
            ├── org-img
        ├── rescuenet-labeled.txt
        ├── rescuenet-unlabeled.txt
        ├── rescuenet-val.txt
```

You may also need to modify the data file path and other hyperparameter in "config_floodnet.yaml" and "config_rescuenet.yaml".

To train S4MC on RescueNet and FloodNet:
```
nohup torchrun --nproc_per_node=6 train_semi.py --config config_floodnet.yaml --seed 42 --name floodnet

nohup torchrun --nproc_per_node=8 train_semi.py --config config_rescuenet.yaml --seed 42 --name rescuenet
```

For evaluation, run
```
python eval-floodnet.py --config config_floodnet.yaml --ckpt checkpoints/floodnet/ckpt_best.pth --save_dir results/eval_floodnet

python eval-rescuenet.py --config config_rescuenet.yaml --ckpt checkpoints/rescuenet/ckpt_best.pth --save_dir results/eval_rescuenet
```

## Dual Teacher

You need to create a new conda environment using the provided "environment.yml".
There is a known issue that some time you may encourter some version error even with this provided yml file.
A working version for me are:
* Python 3.8.20
* PyTorch 1.10.1+cu113
* Cuda 11.3
* CuDNN 8.2
* TorchVision 0.11.2+cu113
* OpenCV 4.11.0
* MMCV 1.3.17
* MMSegmentation 0.11.0+868e6e7

After installing all the packages, you need to run "cd Dual-Teacher && pip install -e . --user"

You may also need the headless opencv installation.

You need to organized the dataset like following:
```
├── ./data
    ├── floodnet
        ├── annotations
            ├── train-label-img-l
            ├── train-label-img-u
            ├── val-label-img
        ├── images
            ├── train-org-img-l
            ├── train-org-img-u
            ├── val-org-img
    ├── rescuenet
        ├── annotations
            ├── train-label-img-l
            ├── train-label-img-u
            ├── val-label-img
        ├── images
            ├── train-org-img-l
            ├── train-org-img-u
            ├── val-org-img
```

"floodnet-organized.py" and "rescuenet-organized.py" may be helpful for you to move the data files.

You may also need to modify the following config files:
* local_configs/segformer/B1/segformer.b1.512x512.floodnet.160k.py
* local_configs/segformer/B1/segformer.b1.512x512.rescuenet.160k.py
* local_configs/_base_/floodnet.py
* local_configs/_base_/rescuenet.py

If your GPU has less memory, you may need to modify the batch_size in "tools/train-flood.py" and "tools/train-rescue.py"


For training, run:
```
nohup python -u -m torch.distributed.launch   --nproc_per_node=8   --master_port=1777   tools/train-flood.py   --ddp   --backbone mit_b1   --save_path logs/floodnet_run   > logs/train_$(date +%m%d_%H%M%S).log 2>&1 &

nohup python -u -m torch.distributed.launch   --nproc_per_node=4   --master_port=1777   tools/train-rescue.py   --ddp   --backbone mit_b1   --save_path logs/rescuenet_run   > logs/train_$(date +%m%d_%H%M%S).log 2>&1 &
```

For evaluation(calculating metrics), run:
```
python tools/eval-floodnet.py \
  --val-img-dir data/floodnet/images/val-org-img \
  --val-mask-dir data/floodnet/annotations/val-label-img \
  --checkpoint FloodNet_Best_weights.pth \
  --save-dir ./floodnet_predictions_750

python tools/eval-rescuenet.py \
  --val-img-dir data/rescuenet/images/val-org-img \
  --val-mask-dir data/rescuenet/annotations/val-label-img \
  --checkpoint RescueNet_Best_weights.pth \
  --save-dir ./rescuenet_predictions_750
```

For generating predicted segmentation masks:
```
python tools/eval-dual-teacher-floodnet.py   --val-img-dir data/floodnet/images/val-org-img   --val-mask-dir data/floodnet/annotations/val-label-img   --checkpoint FloodNet_Best_weights.pth   --save-dir ./floodnet_predictions

python tools/eval-dual-teacher-rescuenet.py   --val-img-dir data/rescuenet/images/val-org-img   --val-mask-dir data/rescuenet/annotations/val-label-img   --checkpoint RescueNet_Best_weights.pth   --save-dir ./rescuenet_predictions
```

## ClassMix

## UniMatch
