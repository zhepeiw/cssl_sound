#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_vggsound.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 23.03.2022
# Last Modified Date: 23.03.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>


import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import pdb


def prepare_split_vggsound_csv(
    root_dir,
    output_dir,
    task_classes,
    train_split=0.8,
    seed=1234,
):
    # collect all classes
    class_names = sorted([os.path.basename(p) for p in glob.glob(os.path.join(root_dir,  'vgg-sound-16k', 'train', '*'))])
    all_paths = {
        name: glob.glob(os.path.join(root_dir, 'vgg-sound-16k/train', name, '*.ogg')) for name in class_names
    }
    all_paths = {
        name: paths for name, paths in all_paths.items() if len(paths) > 10
    }
    class_names = sorted(list(all_paths.keys()))
    name2idx = {name:i for i, name in enumerate(class_names)}

    # train/valid splits
    rng = np.random.default_rng(seed)
    for _, v in all_paths.items():
        rng.shuffle(v)
    train_paths = {
        k:v[:int(len(v)*0.8)] for k, v in all_paths.items()
    }
    valid_paths = {
        k:v[int(len(v)*0.8):] for k, v in all_paths.items()
    }
    test_paths = {
        name: sorted(glob.glob(os.path.join(root_dir, 'vgg-sound-16k/test', name, '*.ogg'))) for name in class_names
    }
    test_paths = {
        name: paths for name, paths in test_paths.items() if len(paths) > 10
    }
    assert len(test_paths) == len(all_paths)

    num_tasks = len(task_classes)
    # class to task mapping
    class2task = {}
    for i, task in enumerate(task_classes):
        for cls in task:
            class2task[cls] = i
    os.makedirs(output_dir, exist_ok=True)

    # splitting by task
    train_splits = {i:[] for i in range(num_tasks)}
    valid_splits = {i:[] for i in range(num_tasks)}
    test_splits = {i:[] for i in range(num_tasks)}

    for name, paths in train_paths.items():
        for fname in paths:
            class_id = name2idx[name]
            train_splits[class2task[class_id]].append((fname, class_id, name))
    for name, paths in valid_paths.items():
        for fname in paths:
            class_id = name2idx[name]
            valid_splits[class2task[class_id]].append((fname, class_id, name))
    for name, paths in test_paths.items():
        for fname in paths:
            class_id = name2idx[name]
            test_splits[class2task[class_id]].append((fname, class_id, name))

    # training csv for each task
    for split, split_data in zip(['train', 'valid', 'test'], [train_splits, valid_splits, test_splits]):
        for task_idx in range(num_tasks):
            data = [{
                'ID': i,
                'wav_path': e[0],
                'class_idx': e[1],
                'class_name': e[2]
            } for i, e in enumerate(split_data[task_idx])]
            df = pd.DataFrame(data)
            task_path = os.path.join(
                output_dir,
                '{}_task{}_raw.csv'.format(split, task_idx)
            )
            df.to_csv(task_path, index=False)


def prepare_vggsound_label_encoder(
    root_dir,
    output_dir,
):
    class_names = sorted([os.path.basename(p) for p in glob.glob(os.path.join(root_dir,  'vgg-sound', 'train', '*'))])
    all_paths = {
        name: glob.glob(os.path.join(root_dir, 'vgg-sound-16k/train', name, '*.ogg')) for name in class_names
    }
    all_paths = {
        name: paths for name, paths in all_paths.items() if len(paths) > 10
    }
    class_names = sorted(list(all_paths.keys()))
    from speechbrain.dataio.encoder import CategoricalEncoder
    label_encoder = CategoricalEncoder()
    label_encoder.load_or_create(
        os.path.join(output_dir, 'label_encoder_vggsound_ordered.txt'),
        from_iterables=[class_names],
    )


if __name__ == '__main__':
    #  import numpy as np
    #  task_classes = np.array_split(np.arange(308), 6)
    #  task_classes = [tuple(e) for e in task_classes]
    #  hparams = {
    #      'root_dir': '/mnt/data/Sound Sets/VGG-Sound',
    #      'output_dir': './csvs',
    #      'task_classes': task_classes,
    #      'train_split': 0.8,
    #  }
    #  prepare_split_vggsound_csv(**hparams)
    prepare_vggsound_label_encoder(
        '/mnt/data/Sound Sets/VGG-Sound',
        './dataset',
    )
