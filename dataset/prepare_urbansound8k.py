#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_urbansound8k.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 26.01.2022
# Last Modified Date: 26.01.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

import os
import glob
import csv
from tqdm import tqdm
import pandas as pd
import pdb


def prepare_split_urbansound8k_csv(
    root_dir,
    output_dir,
    task_classes,
    train_folds,
    valid_folds,
    test_folds,
):
    '''
        preparing stagewise csv files for training, validation and test

        args:
        ----
            root_dir: directory containing the 'audio' and 'metadata' directories
            output_dir: directory to save the csv files
            num_stages: int, number of CL stages

    '''
    num_tasks = len(task_classes)
    # class to task mapping
    class2task = {}
    for i, task in enumerate(task_classes):
        for cls in task:
            class2task[cls] = i
    os.makedirs(output_dir, exist_ok=True)
    #  # training files
    #  train_fold_str = ''.join([str(fold) for fold in train_folds])
    #  train_files = glob.glob(os.path.join(root_dir, 'audio', 'fold[{}]'.format(train_fold_str), '*.wav'))
    meta_df = pd.read_csv(os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv'))
    train_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]
    valid_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]
    test_splits = {i:[] for i in range(num_tasks)}  # task_id: [paths]
    for idx, row in tqdm(meta_df.iterrows()):
        fold = row['fold']
        class_id = row['classID']
        class_name = row['class']
        fname = os.path.join(root_dir, 'audio', 'fold{}'.format(fold), row['slice_file_name'])
        if class_id in class2task:
            if fold in train_folds:
                train_splits[class2task[class_id]].append((fname, class_id, class_name))
            if fold in valid_folds:
                valid_splits[class2task[class_id]].append((fname, class_id, class_name))
            if fold in test_folds:
                test_splits[class2task[class_id]].append((fname, class_id, class_name))
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


if __name__ == '__main__':
    hparams = {
        'root_dir': '/mnt/data/Sound Sets/UrbanSound8K/UrbanSound8K',
        'output_dir': './csvs',
        'task_classes': [(2*i, 2*i+1) for i in range(5)],
        'train_folds': [0, 1, 2, 3, 4, 5, 6, 7],
        'valid_folds': [8],
        'test_folds': [9],
    }
    prepare_split_urbansound8k_csv(**hparams)
