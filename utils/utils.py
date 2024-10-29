import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)

def set_random_seed(seed=None):
    if seed is not None:
        # raise NotImplementedError('Fixing seed has not yet been implemented.')
        print('Set random seed as', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
        # TODO: this leads to performance degradation, while the result can not be repeated
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.benchmark = True

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def sum(self):
        return sum(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def save_curve(args, save_dir, training_losses, test_clean_losses, 
               overall_accs, many_accs, median_accs, low_accs, f1s):
    plt.figure()
    plt.plot(training_losses, 'b', label='training_losses')
    plt.plot(test_clean_losses, 'g', label='test_clean_losses')
    plt.grid()
    plt.legend()
    plt.savefig(osp.join(save_dir, 'losses.png'))
    plt.close()

    plt.plot(overall_accs, 'm', label='overall_accs')
    if args.imbalance_ratio < 1:
        plt.plot(many_accs, 'r', label='many_accs')
        plt.plot(median_accs, 'g', label='median_accs')
        plt.plot(low_accs, 'b', label='low_accs')
    plt.grid()
    plt.legend()
    plt.savefig(osp.join(save_dir, 'test_accs.png'))
    plt.close()

    plt.plot(f1s, 'm', label='f1s')
    plt.grid()
    plt.legend()
    plt.savefig(osp.join(save_dir, 'test_f1s.png'))
    plt.close()


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
