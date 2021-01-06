"""
forest Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import logging
import datasets.uniform as uniform
import datasets.forest_labels as forest_labels
import json
from config import cfg

trainid_to_name = forest_labels.trainId2name
id_to_trainid = forest_labels.label2trainid
num_classes = 21
ignore_label = 255
root = cfg.DATASET.FOREST_DIR_SEMANTIC
aug_root = cfg.DATASET.KITTI_AUG_DIR

palette = [29, 28, 33, 208, 235, 160, 43, 237, 21, 217, 240, 17, 186, 24, 65, 237, 9, 28,
           235, 45, 98, 20, 99, 143, 157, 199, 194, 237, 61, 55, 32, 39, 232, 37, 193, 245,
           132, 143, 127, 25, 151, 209, 83, 90, 169, 158, 163, 62, 182, 55, 127, 101, 28, 173,
           162, 168, 104, 162, 135, 176, 45, 149, 238]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def get_train_val(cv_split, all_items):
    # 90/10 train/val split, three random splits for cross validation
    # val_0 = [1,5,11,29,35,49,57,68,72,82,93,115,119,130,145,154,156,167,169,189,198]
    # val_1 = [0,12,24,31,42,50,63,71,84,96,101,112,121,133,141,155,164,171,187,191,197]
    # val_2 = [3,6,13,21,41,54,61,73,88,91,110,121,126,131,142,149,150,163,173,183,199]


    # 90/10 train/val split
    train_set = []
    val_set = []

    num_all_items = len(all_items)

    if cv_split == 0:
        for i in range(num_all_items):
            if i%10 == 0:
                val_set.append(all_items[i])
            else:
                train_set.append(all_items[i])
    # elif cv_split == 1:
    #     for i in range(num_all_items):
    #         if i in val_1:
    #             val_set.append(all_items[i])
    #         else:
    #             train_set.append(all_items[i])
    # elif cv_split == 2:
    #     for i in range(num_all_items):
    #         if i in val_2:
    #             val_set.append(all_items[i])
    #         else:
    #             train_set.append(all_items[i])
    else:
        logging.info('Unknown cv_split {}'.format(cv_split))
        sys.exit()

    return train_set, val_set

def make_dataset(quality, mode, maxSkip=0, cv_split=0, hardnm=0):
    items = []
    all_items = []
    aug_items = []

    assert quality == 'semantic'
    assert mode in ['train', 'val', 'trainval']
    # note that train and val are randomly determined, no official split

    img_dir_name = "training"
    img_path = os.path.join(root, img_dir_name, 'images')
    mask_path = os.path.join(root, img_dir_name, 'labels_id')

    c_items = os.listdir(img_path)
    c_items.sort()

    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        all_items.append(item)
    logging.info('forest has a total of {} images'.format(len(all_items)))

    # split into train/val
    train_set, val_set = get_train_val(cv_split, all_items)
    #print(train_set)

    if mode == 'train':
        items = train_set
    elif mode == 'val':
        items = val_set
    elif mode == 'trainval':
        items = train_set + val_set
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('forest-{}: {} images'.format(mode, len(items)))

    return items, aug_items

def make_test_dataset(quality, mode, maxSkip=0, cv_split=0):
    items = []
    assert quality == 'semantic'
    assert mode == 'test'

    img_dir_name = "testing"
    img_path = os.path.join(root, img_dir_name, 'images')
    mask_path = os.path.join(root, img_dir_name, 'labels_id')

    c_items = os.listdir(img_path)
    c_items.sort()
    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        items.append(item)
    logging.info('forest has a total of {} test images'.format(len(items)))

    return items, []

class FOREST_Semantic(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False,
                 cv_split=None, scf=None, hardnm=0):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.scf = scf
        self.hardnm = hardnm

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0

        if self.mode == 'test':
            self.imgs, _ = make_test_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split)
        else:
            self.imgs, _ = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split, hardnm=self.hardnm)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            if self.scf:
                json_fn = 'forest_tile{}_cv{}_scf.json'.format(self.class_uniform_tile, self.cv_split)
            else:
                json_fn = 'forest_tile{}_cv{}_{}_hardnm{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode, self.hardnm)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                if self.scf:
                    self.centroids = kitti_uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=id_to_trainid,
                        tile_size=class_uniform_tile)
                else:
                    self.centroids = uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=id_to_trainid,
                        tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

        self.build_epoch()

    def build_epoch(self, cut=False):
        if self.class_uniform_pct > 0:
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem

        if self.mode == 'test':
            img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        else:
            img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # forest scale correction factor
        if self.mode == 'train' or self.mode == 'trainval':
            if self.scf:
                width, height = img.size
                img = img.resize((width*2, height*2), Image.BICUBIC)
                mask = mask.resize((width*2, height*2), Image.NEAREST)
        elif self.mode == 'val':
            width, height = 640, 480
            img = img.resize((width, height), Image.BICUBIC)
            mask = mask.resize((width, height), Image.NEAREST)
        elif self.mode == 'test':
            img_keepsize = img.copy()
            width, height = 640, 480
            img = img.resize((width, height), Image.BICUBIC)
            mask = mask.resize((width, height), Image.NEAREST)
        else:
            logging.info('Unknown mode {}'.format(mode))
            sys.exit()

        # if self.mode != 'test':
        #     mask = np.array(mask)
        #     mask_copy = mask.copy()

        #     for k, v in id_to_trainid.items():
        #         mask_copy[mask == k] = v
        #     mask = Image.fromarray(mask_copy.astype(np.uint8))
        mask = np.array(mask)
        mask_copy = mask.copy()

        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)
