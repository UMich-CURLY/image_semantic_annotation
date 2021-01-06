"""
Test Script
"""

import os
import logging
import sys
import argparse
import re
import queue
import threading
from math import ceil
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import PIL

from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import transforms.transforms as extended_transforms

from config import assert_and_infer_cfg
from datasets import cityscapes, kitti, kitti_semantic, kitti_trav
from optimizer import restore_snapshot

from utils.my_data_parallel import MyDataParallel
from utils.misc import fast_hist, save_log, per_class_iu, evaluate_eval_for_inference

import network

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), '../'))

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dump_images', action='store_true', default=False)
parser.add_argument('--arch', type=str, default='', required=True)
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, video_folder')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--cv_split', type=int, default=None)
parser.add_argument('--mode', type=str, default='semantic')
parser.add_argument('--dataset_cls', type=str, default='cityscapes', help='cityscapes')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Dataset Location')
parser.add_argument('--snapshot', required=True, type=str, default='')
parser.add_argument('--snapshot2', required=True, type=str, default='')

args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
args.apex = False  # No support for apex eval
cudnn.benchmark = False
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


def get_net():
    """
    Get Network for evaluation
    """
    logging.info('Load model file: %s', args.snapshot)
    net = network.get_net_ori(args, criterion=None)

    net = torch.nn.DataParallel(net).cuda()
    net, _ = restore_snapshot(net, optimizer=None,
                              snapshot=args.snapshot, snapshot2=args.snapshot2, restore_optimizer_bool=False)
    net.eval()
    return net

def setup_loader():
    """
    Setup Data Loaders
    """
    val_input_transform = transforms.ToTensor()
    target_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'cityscapes':
        args.dataset_cls = cityscapes
        eval_mode_pooling = False
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode_pooling = True
            eval_scales = args.scales
        test_set = args.dataset_cls.CityScapes(args.mode, args.split,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=args.cv_split,
                                               eval_mode=eval_mode_pooling,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               )
    elif args.dataset == 'kitti':
        args.dataset_cls = kitti
        test_set = args.dataset_cls.KITTI(args.mode, args.split,
                                         transform=val_input_transform,
                                         target_transform=target_transform,
                                         cv_split=args.cv_split)

    elif args.dataset == 'kitti_trav':
        args.dataset_cls = kitti_trav
        test_set = args.dataset_cls.KITTI_trav(args.mode, args.split,
                                         transform=val_input_transform,
                                         target_transform=target_transform,
                                         cv_split=args.cv_split)

    elif args.dataset == 'kitti_semantic':
        args.dataset_cls = kitti_semantic
        test_set = args.dataset_cls.KITTI_Semantic(args.mode, args.split,
                                         transform=val_input_transform,
                                         target_transform=target_transform,
                                         cv_split=args.cv_split)
    else:
        raise NameError('-------------Not Supported Currently-------------')

#    if args.split_count > 1:
#        test_set.split_dataset(args.split_index, args.split_count)

    batch_size = 1

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1,
                             shuffle=False, pin_memory=False, drop_last=False)

    return test_loader

def inf(imgs, img_names, gt, net, scales, pbar, base_img):

    ######################################################################
    # Run inference
    ######################################################################

    img_name = img_names[0]
    pred_img_name_semantic = '{}/{}_semantic.png'.format('./eval/', img_name)
    pred_img_name_trav = '{}/{}_trav.png'.format('./eval/', img_name)
    col_img_name = '{}/{}_color.png'.format('./eval/', img_name)
    compose_img_name = '{}/{}_compose.png'.format('./eval/', img_name)
    to_pil = transforms.ToPILImage()
    img = to_pil(imgs[0])
    prediction1, prediction2 = net(imgs.cuda())#,task='traversability'

    prediction1 = np.argmax(prediction1.squeeze().detach().cpu().numpy(), axis=0)
    prediction2 = np.argmax(prediction2.squeeze().detach().cpu().numpy(), axis=0)

    ######################################################################
    # Dump Images
    ######################################################################

    colorized = args.dataset_cls.colorize_mask(prediction1)
    colorized.save(col_img_name)
    blend = Image.blend(img.convert("RGBA"), colorized.convert("RGBA"), 0.5)
    blend.save(compose_img_name)

    label_out = np.zeros_like(prediction1)
    for label_id, train_id in args.dataset_cls.id_to_trainid.items():
        label_out[np.where(prediction1 == train_id)] = label_id
    cv2.imwrite(pred_img_name_semantic, label_out)

    label_out = np.zeros_like(prediction2)
    for label_id, train_id in args.dataset_cls.id_to_trainid.items():
        label_out[np.where(prediction1 == train_id)] = label_id
    cv2.imwrite(pred_img_name_trav, label_out)

def main():
    """
    Main Function
    """

    # Set up network, loader, inference mode
    metrics = args.dataset != 'video_folder'
    if args.dataset == 'kitti' and args.split == 'test':
        metrics = False
    test_loader = setup_loader()

    net = get_net()

    # Run Inference!
    pbar = tqdm(test_loader, desc='eval {}'.format(args.split), smoothing=1.0)
    for iteration, data in enumerate(pbar):

        base_img = None
        imgs, gt, img_names = data
        inf(imgs, img_names, gt, net, [1.0], pbar, base_img)




if __name__ == '__main__':
    main()
