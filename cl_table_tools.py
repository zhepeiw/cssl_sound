import torch
import numpy as np
import argparse
import pdb


def compute_cl_statistics(table):
    '''
        table[i, j] is the accuracy of task j after being trained on task i
    '''
    avg_acc = np.mean(table[-1])
    forgetting = np.max(table - table[-1][None, :], axis=0)
    avg_fgt = forgetting[:-1].mean()
    avg_seen_acc = np.mean(table[np.tril_indices(table.shape[0], -1)])  # off diagonal lower triangle, evaluating all past tasks
    avg_full_transfer_acc = np.mean(table[np.triu_indices(table.shape[0], 1)])
    report = {
        'avg_acc': avg_acc,
        'avg_fgt': avg_fgt,
        'avg_seen': avg_seen_acc,
        'avg_full': avg_full_transfer_acc,
    }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str,
                        help='Path to the saved cl table')
    args = parser.parse_args()
    table = torch.load(args.path)
    print(compute_cl_statistics(table))
