#!/usr/bin/env bash

    # Example on KITTI, fine tune
     python -m torch.distributed.launch --nproc_per_node=2 train_trav.py \
        --dataset kitti_trav \
        --cv 2 \
        --arch network.deepv3.DeepWV3Plus_trav \
        --snapshot ./pretrained_models/kitti_trav.pth \
        --snapshot2 ./pretrained_models/kitti_semantic.pth \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 300 \
        --lr 0.001 \
        --lr_schedule poly \
        --poly_exp 1.0 \
        --syncbn \
        --sgd \
        --crop_size 360 \
        --scale_min 1.0 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --max_epoch 90 \
        --img_wt_loss \
        --wt_bound 1.0 \
        --bs_mult 2 \
        --apex \
        --exp kitti_ft \
        --ckpt ./logs/ \
        --tb_path ./logs/


