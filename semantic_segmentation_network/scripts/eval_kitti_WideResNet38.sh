#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 eval.py \
	--dataset kitti_semantic \
    --arch network.deepv3.DeepWV3Plus_semantic \
    --inference_mode sliding \
    --scales 1.0 \
    --split train \
    --cv_split 2 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
