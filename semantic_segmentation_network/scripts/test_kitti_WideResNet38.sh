#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 test.py \
    --dataset kitti \
    --arch network.deepv3.DeepWV3Plus \
    --mode semantic \
    --split train \
    --cv_split 0 \
    --dump_images \
    --snapshot ${1} \
    --snapshot2 ${2}
