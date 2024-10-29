# Copyright (c) Alibaba, Inc. and its affiliates.
'''
Codes adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/test.py
which uses Apache-2.0 license.
'''
import os, argparse, time
from contextlib import ExitStack
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from datasets.SCOODBenchmarkDataset import SCOODDataset

from utils.common import build_dataset, build_model, build_prior, build_plain_train_loader
from utils.utils import AverageMeter
from utils.ltr_metrics import shot_acc, shot_ood
from models.base import get_ood_scores
from models.feat_pool import IDFeatPool
# from utils.ood_metrics import get_ood_scores

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, thres = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, thres

## Test on CIFAR:
def val_cifar():
    '''
    Evaluate ID acc and OOD detection on CIFAR10/100
    '''
    model.eval()
    ts = time.time()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    probs_list = []

    fp = open(f'{args.ckpt_path}/res.txt', 'w+')

    with ExitStack() as stack:
        if all(x not in args.ood_metric for x in ['odin', 'gradnorm']):
            stack.enter_context(torch.no_grad())

        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_ood_scores(model, images, args.ood_metric, adjustments)
            # probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            # probs_list.append(probs.max(dim=1).values.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    # print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    # in_probs = np.concatenate(probs_list, axis=0)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)
    fp.write('\n===ID Accuracy===\n')
    fp.write('ACC: %.4f (%.4f, %.4f, %.4f)\n' % (test_acc, many_acc, median_acc, low_acc))
    fp.flush()

    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    ood_res, ood_stat = [], []
    for dout in ['texture', 'svhn', 'cifar', 'tin', 'lsun', 'places365']:
        if dout == 'cifar':
            # continue
            if args.dataset == 'cifar10':
                dout = 'cifar100'
            elif args.dataset == 'cifar100':
                dout = 'cifar10'
        ood_set = SCOODDataset(osp.join(args.data_root_path, 'SCOOD'), id_name=args.dataset, ood_name=dout, transform=test_transform)
        ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)
        # confidence distribution of correct samples:
        ood_score_list, sc_labels_list = [], []
        ood_pred_list = []
        with ExitStack() as stack:
            if all(x not in args.ood_metric for x in ['odin', 'gradnorm']):
                stack.enter_context(torch.no_grad())

            for images, sc_labels in ood_loader:
                images, sc_labels = images.cuda(), sc_labels.cuda()
                logits, scores = get_ood_scores(model, images, args.ood_metric, adjustments)
                # probs = F.softmax(logits, dim=1)
                pred = logits.data.max(1)[1]
                # append loss:
                ood_score_list.append(scores.detach().cpu().numpy())
                sc_labels_list.append(sc_labels.detach().cpu().numpy())
                ood_pred_list.append(pred.detach().cpu().numpy())
        ood_scores = np.concatenate(ood_score_list, axis=0)
        sc_labels = np.concatenate(sc_labels_list, axis=0)
        ood_preds = np.concatenate(ood_pred_list, axis=0)

        # move some elements in ood_scores to in_scores:
        # print('in_scores:', in_scores.shape)
        # print('ood_scores:', ood_scores.shape)
        fake_ood_scores = ood_scores[sc_labels>=0]
        real_ood_scores = ood_scores[sc_labels<0]
        real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)
        real_ood_preds = ood_preds[sc_labels<0]
        # print('fake_ood_scores:', fake_ood_scores.shape)
        # print('real_in_scores:', real_in_scores.shape)
        # print('real_ood_scores:', real_ood_scores.shape)

        auroc, aupr, fpr95, thres = get_measures(real_ood_scores, real_in_scores)
        ood_res.append([auroc, aupr, fpr95])

        # print:
        ood_detection_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
        print(ood_detection_str)
        fp.write('\n===%s===\n' % dout)
        fp.write(ood_detection_str + '\n')
        fp.flush()
        
        false_id_labels = in_labels[in_scores >= thres]
        false_id_preds = in_preds[in_scores >= thres]
        false_ood_preds = real_ood_preds[real_ood_scores < thres]
        ood_stat.append([*shot_ood(false_id_labels, img_num_per_cls), 
                         *shot_ood(false_id_preds, img_num_per_cls), 
                         *shot_ood(false_ood_preds, img_num_per_cls)])
        

    ood_res, ood_stat = np.array(ood_res, dtype=np.float32), np.array(ood_stat, dtype=np.int32)
    ood_detection_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % tuple(ood_res.mean(axis=0).tolist())
    fp.write('\n===Total===\n')
    fp.write(ood_detection_str + '\n')
    fp.flush()
    fp.close()
    print('\nTotal:')
    # print('Sta:', ood_stat.sum(axis=0), ood_stat.sum())
    print('OOD:', ood_detection_str)
    print('ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc))
    with open(f'{args.ckpt_path}/res.csv', 'w+') as f:
        f.write('%.2f\n' % (test_acc*100))
        f.write('%.2f\n%.2f\n' % tuple((ood_res.mean(axis=0)*100)[[0,2]].tolist()))
        f.write(('\n'.join(['%d'] * ood_stat.shape[1])) % tuple(ood_stat.sum(axis=0).tolist()))


def val_imagenet():
    '''
    Evaluate ID acc and OOD detection on ImageNet-1k
    '''
    model.eval()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []

    fp = open(f'{args.ckpt_path}/res.txt', 'w+')

    with ExitStack() as stack:
        if all(x not in args.ood_metric for x in ['odin', 'gradnorm']):
            stack.enter_context(torch.no_grad())

        for images, targets in tqdm(test_loader, desc='in', total=len(test_loader)):
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_ood_scores(model, images, args.ood_metric, adjustments)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())

    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)
    in_clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    fp.write('\n===ID Accuracy===\n')
    fp.write(in_clean_str + '\n')
    fp.flush()

    if args.dataset == 'imagenet':
        dout = 'imagenet10k'
        dout_path = 'imagenet/ood_test_1k'
    elif args.dataset == 'waterbird':
        dout = 'spurious_ood'
        dout_path = 'waterbird_LT/spuriuos_ood'
    else:
        raise NotImplementedError
    test_transform = test_loader.dataset.transform
    ood_set = ImageFolder(osp.join(args.data_root_path, dout_path), transform=test_transform)
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    dout_str = 'Dout is %s with %d images' % (dout, len(ood_set))
    print(dout_str)

    # confidence distribution of correct samples:
    ood_score_list, ood_pred_list = [], []
    with ExitStack() as stack:
        if all(x not in args.ood_metric for x in ['odin', 'gradnorm']):
            stack.enter_context(torch.no_grad())

        for images, _ in tqdm(ood_loader, desc='out', total=len(ood_loader)):
            images = images.cuda()
            logits, scores = get_ood_scores(model, images, args.ood_metric, adjustments)
            pred = logits.data.max(1)[1]
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
            ood_pred_list.append(pred.detach().cpu().numpy())
    ood_scores = np.concatenate(ood_score_list, axis=0)
    ood_preds = np.concatenate(ood_pred_list, axis=0)

    auroc, aupr, fpr95, thres = get_measures(ood_scores, in_scores)
    ood_res = np.array([auroc, aupr, fpr95])

    # print:
    ood_detection_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
    fp.write('\n===%s===\n' % dout)
    fp.write(ood_detection_str + '\n')
    fp.flush()
    
    false_id_labels = in_labels[in_scores >= thres]
    false_id_preds = in_preds[in_scores >= thres]
    false_ood_preds = ood_preds[ood_scores < thres]
    ood_stat = np.array([*shot_ood(false_id_labels, img_num_per_cls), 
                         *shot_ood(false_id_preds, img_num_per_cls), 
                         *shot_ood(false_ood_preds, img_num_per_cls)])
    
    fp.close()
    
    # print('Sta:', ood_stat, ood_stat.sum())
    print('OOD:', ood_detection_str)
    print('ACC:', in_clean_str)
    with open(f'{args.ckpt_path}/res.csv', 'w+') as f:
        f.write('%.2f\n' % (test_acc*100))
        f.write('%.2f\n%.2f\n' % tuple((ood_res*100)[[0,2]].tolist()))
        f.write(('\n'.join(['%d'] * len(ood_stat))) % tuple(ood_stat.tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--num_workers', type=int, default=4)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'waterbird'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='./data/', help='Where you save all your datasets.')
    parser.add_argument('--id_class_number', type=int, default=1000, help='for ImageNet subset')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'WRN40', 'ResNet50'], help='which model to use')
    # 
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
    parser.add_argument('--logit_adjust', '--tau', default=0., type=float)
    parser.add_argument('--ood_metric', default='msp', choices=['msp', 'bkg_c', 'energy', 'bin_disc', 'mc_disc', 'rp_msp', 'gradnorm', 'rp_gradnorm', 'rp_gradnorm', 'rw_energy', 'maha',
                                                                'ada_bin_disc', 'ada_msp', 'ada_energy', 'ada_maha', 'ada_gradnorm'], help='OOD training metric')
    parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
    #
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--ckpt', default='latest', choices=['latest', 'best'])
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    args = parser.parse_args()
    # print(args)
    args.batch_size = args.test_batch_size

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    num_classes, test_loader, img_num_per_cls_and_ood = build_dataset(args, 1, is_training=False)
    img_num_per_cls = img_num_per_cls_and_ood[:num_classes]

    model, num_outputs = build_model(args, num_classes, 'cuda', args.gpu, return_features=False, is_training=False)

    adjustments = build_prior(args, model, img_num_per_cls, num_classes, num_outputs, 'cuda')

    # load model:
    if args.ckpt == 'latest':
        ckpt = torch.load(osp.join(args.ckpt_path, 'latest.pth'))['model']
    else:
        ckpt = torch.load(osp.join(args.ckpt_path, 'best_clean_acc.pth'))
    if list(ckpt.keys())[0].startswith('module.'):  # compatible with pretrained PASCL
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}  
    model.load_state_dict(ckpt, strict=False)
    model.requires_grad_(False)

    if 'maha' in args.ood_metric:
        model.id_feat_pool = IDFeatPool(num_classes, sample_num=max(img_num_per_cls), 
                                        feat_dim=model.penultimate_layer_dim, device='cuda')
        plain_train_loader = build_plain_train_loader(args)
        model.eval()
        id_feat_path = f'{args.ckpt_path}/id_feats.pth'
        if not osp.exists(id_feat_path):  # lower priority
            id_feat_path = f'{args.ckpt_path}/id_feat_plain.pt'
        if osp.exists(id_feat_path):
            model.id_feat_pool.load(id_feat_path)
        else:
            for images, labels in tqdm(plain_train_loader, desc='Doing statistics'):
                images, labels = images.cuda(), labels.cuda()
                _, p4 = model(images, return_features=True)
                model.id_feat_pool.update(p4, labels)
            model.id_feat_pool.save(id_feat_path)

    with torch.no_grad():
        if 'cifar' in args.dataset:
            val_cifar()
        else:
            val_imagenet()