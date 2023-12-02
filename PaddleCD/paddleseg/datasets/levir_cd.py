# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import os
import numpy as np
import random
from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class LevirCD(Dataset):
    NUM_CLASSES = 2
    IMG_CHANNELS = 3
    IGNORE_INDEX = 255

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        # 图像及标签的路径
        img1_dir = os.path.join(self.dataset_root, 'A')
        img2_dir = os.path.join(self.dataset_root, 'B')
        label_dir = os.path.join(self.dataset_root, 'target')
        if dataset_root is None or not os.path.isdir(
                dataset_root) or not os.path.isdir(
                    img1_dir) or not os.path.isdir(img2_dir
                    ) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        # 对文件排序
        files = sorted(os.listdir(label_dir))

        self.data_list = []

        # 把图像和标签对应
        for item in files:
            self.data_list.append([
                os.path.join(img1_dir, item),
                os.path.join(img2_dir, item),
                os.path.join(label_dir, item)
            ])

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image1_path, image2_path, label_path = self.data_list[idx]
        data['img1'] = image1_path
        data['img2'] = image2_path
        data['label'] = label_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []
        if self.mode == 'val':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]

        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)

        # 数据增强，降低时间维度带来的影响
        data["img"] = np.concatenate([data["img1"], data["img2"]], axis=0) if random.random() > 0.5 \
                      else np.concatenate([data["img2"], data["img1"]], axis=0)
        # print(data['label'].shape,  data['label'].max(),  data['label'].min(),  data['label'].dtype)
        return data
    
    def __len__(self):
        return len(self.data_list)
