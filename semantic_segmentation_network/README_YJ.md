0. Docker run

$ sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic_segmentation_train:/home/workspace -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_trav:/home/dataset/trav -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_semantics:/home/dataset/semantic nvidia-segmentation:latest bash

$ sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic_segmentation_train:/home/workspace -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_trav:/home/dataset nvidia-segmentation:latest bash

1. Train

[Train all] 
$ ./scripts/train_kitti_WideResNet38.sh

[Train semantic] 
$ ./scripts/train_kitti_WideResNet38_semantic.sh

[Train traversability] 
$ ./scripts/train_kitti_WideResNet38_trav.sh

2. Evaluation

[Evalutate semantic]
$ ./script/eval_kitti_WideResNet38.sh ./ckpts/kitti_semantic_final.pth ./eval/

[Evalutate traversability]
$ ./script/eval_kitti_WideResNet38.sh ./ckpts/kitti_trav_final.pth ./eval/

3. Test

$ ./script/test_kitti_WideResNet38.sh ./ckpts/kitti_trav_final.pth ./ckpts/kitti_semantic_final.pth

