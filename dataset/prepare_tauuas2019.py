#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_tauuas2019.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 26.03.2022
# Last Modified Date: 30.03.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

import os
from tqdm import tqdm
import pandas as pd
import pdb


def prepare_split_tauuas2019_csv(
    root_dir,
    output_dir,
    task_classes,
):
    '''
        preparing stagewise csv files for training, validation and test

        args:
        ----
            root_dir: directory containing the 'audio' and 'metadata' directories
            output_dir: directory to save the csv files
            task_classes: list of tuples, each tuple contains the class index for each task
    '''
    train_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_train.csv')
    train_meta = pd.read_csv(train_csv, sep='\t')
    valid_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_test.csv')
    valid_meta = pd.read_csv(valid_csv, sep='\t')
    test_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_evaluate.csv')
    test_meta = pd.read_csv(test_csv, sep='\t')
    class_names = sorted(train_meta['scene_label'].unique().tolist())
    name2idx = {name:i for i, name in enumerate(class_names)}

    num_tasks = len(task_classes)
    # class to task mapping
    class2task = {}
    for i, task in enumerate(task_classes):
        for cls in task:
            class2task[cls] = i
    os.makedirs(output_dir, exist_ok=True)

    train_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]
    valid_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]
    test_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]

    for meta_df, splits in zip([train_meta, valid_meta, test_meta], [train_splits, valid_splits, test_splits]):
        for idx, row in tqdm(meta_df.iterrows()):
            fname = os.path.join(root_dir, row['filename'])
            fname = fname.replace('audio', 'audio_mono_16k')
            if 'scene_label' in meta_df:
                class_name = row['scene_label']
            else:
                class_name = os.path.basename(fname).split('-')[0]
            task_id = class2task[name2idx[class_name]]
            splits[task_id].append((fname, class_name))

    # training csv for each task
    for split, split_data in zip(['train', 'valid', 'test'], [train_splits, valid_splits, test_splits]):
        for task_idx in range(num_tasks):
            data = [{
                'ID': i,
                'wav_path': e[0],
                'class_name': e[1]
            } for i, e in enumerate(split_data[task_idx])]
            df = pd.DataFrame(data)
            task_path = os.path.join(
                output_dir,
                '{}_task{}_raw.csv'.format(split, task_idx)
            )
            df.to_csv(task_path, index=False)


def prepare_tauuas2019_label_encoder(
    root_dir,
    output_dir,
):
    train_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_train.csv')
    train_meta = pd.read_csv(train_csv, sep='\t')
    class_names = sorted(train_meta['scene_label'].unique().tolist())
    from speechbrain.dataio.encoder import CategoricalEncoder
    label_encoder = CategoricalEncoder()
    label_encoder.load_or_create(
        os.path.join(output_dir, 'label_encoder_tauuas2019_ordered.txt'),
        from_iterables=[class_names],
    )


if __name__ == '__main__':
    #  import numpy as np
    #  task_classes = np.array_split(np.arange(10), 5)
    #  task_classes = [tuple(e) for e in task_classes]
    #  hparams = {
    #      'root_dir': '/mnt/data/Sound Sets/TAU-urban-acoustic-scenes-2019-development',
    #      'output_dir': './csvs',
    #      'task_classes': task_classes,
    #  }
    #  prepare_split_tauuas2019_csv(**hparams)
    prepare_tauuas2019_label_encoder(
        '/mnt/data/Sound Sets/TAU-urban-acoustic-scenes-2019-development',
        './dataset',
    )
