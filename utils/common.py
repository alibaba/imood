# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from datasets.tinyimages_300k import TinyImages
from models.base import BaseModel
from models.resnet import ResNet34, ResNet18
from models.resnet_imagenet import ResNet50# , ResNet18
from models.feat_pool import IDFeatPool

from utils.utils import TwoCropTransform, de_parallel


def build_dataset(args, ngpus_per_node, is_training=True):
    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == 'imagenet' or args.dataset == 'waterbird':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = args.id_class_number
        train_set = LT_Dataset(
            osp.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=TwoCropTransform(train_transform), 
            subset_class_idx=np.arange(0,args.id_class_number))
        if is_training:
            test_set = LT_Dataset(
                osp.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_val.txt', transform=test_transform,
                subset_class_idx=np.arange(0,args.id_class_number))
        else:
            test_set = ImageFolder(osp.join(args.data_root_path, 'imagenet', 'val'), transform=test_transform)
    elif args.dataset == 'waterbird':
        num_classes = 2
        train_set = ImageFolder(osp.join(args.data_root_path, 'waterbird_LT', 'train'), transform=TwoCropTransform(train_transform))
        setattr(train_set, 'img_num_per_cls', [363, 3699])
        if is_training:
            test_set = ImageFolder(osp.join(args.data_root_path, 'waterbird_LT', 'val'), transform=test_transform)
        else:
            test_set = ImageFolder(osp.join(args.data_root_path, 'waterbird_LT', 'test'), transform=test_transform)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    
    if is_training:
        if args.ood_aux_dataset in ['TinyImages', 'ImExtra', 'Texture']:
            if args.dataset in ['cifar10', 'cifar100']:
                ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))
            elif args.dataset == 'imagenet':
                ood_set = ImageFolder(osp.join(args.data_root_path, 'imagenet/extra_1k'), transform=train_transform)
            elif args.dataset == 'waterbird':
                ood_set = ImageFolder(osp.join(args.data_root_path, 'texture'), transform=train_transform)
            else:
                raise NotImplementedError(args.dataset, args.ood_aux_dataset)
            
            if args.ddp:
                ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set)
            else:
                ood_sampler = None
            ood_loader = DataLoader(ood_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                        drop_last=True, pin_memory=True, sampler=ood_sampler)
            ood_num = len(ood_set)
            print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
        elif args.ood_aux_dataset in ['VOS', 'NPOS']:
            # sample_num = min(train_set.img_num_per_cls) * 10
            sample_num = max(train_set.img_num_per_cls)
            feat_dim = 512 if 'cifar' in args.dataset else 1024  # TODO: check
            mode = args.ood_aux_dataset
            device = 'cuda:0'
            ood_loader = IDFeatPool(num_classes, sample_num=sample_num, feat_dim=feat_dim, mode=mode, device=device)
            ood_num = num_classes * sample_num
        elif args.ood_aux_dataset in ['CIFAR']:
            if args.dataset == 'cifar10':
                dout = 'cifar100'
            elif args.dataset == 'cifar100':
                dout = 'cifar10'
            ood_set = ImageFolder(osp.join(args.data_root_path, f'SCOOD/data/images/{dout}/test'), transform=train_transform)
            if args.ddp:
                ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set)
            else:
                ood_sampler = None
            ood_loader = DataLoader(ood_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                        drop_last=True, pin_memory=True, sampler=ood_sampler)
            ood_num = len(ood_set)
            print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
        else:
            raise NotImplementedError(f'{args.dataset} v.s. {args.ood_aux_dataset}')

        img_num_per_cls_and_ood = np.array(train_set.img_num_per_cls + [ood_num])

        return num_classes, train_loader, test_loader, ood_loader, train_sampler, img_num_per_cls_and_ood
    else:

        img_num_per_cls_and_ood = np.array(train_set.img_num_per_cls + [args.num_ood_samples])

        return num_classes, test_loader, img_num_per_cls_and_ood


def build_plain_train_loader(args):  # for statistic during test
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = args.id_class_number
        train_set = LT_Dataset(
            osp.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=test_transform, 
            subset_class_idx=np.arange(0,args.id_class_number))
        
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4,
                                drop_last=True, pin_memory=True)
    
    return train_loader



def build_model(args, num_classes, device, gpu_id, return_features=False, is_training=True):
    # model:
    num_outputs = num_classes
    if any(x in args.ood_metric for x in ['bkg_c', 'bin_disc']):
        num_outputs += 1
    elif any(x in args.ood_metric for x in ['mc_disc']):
        num_outputs += 2
    # 'ResNet18', 'ResNet34', or 'ResNet50'
    model: BaseModel = eval(args.model)(num_classes=num_classes, num_outputs=num_outputs, return_features=return_features).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu_id], broadcast_buffers=False,
            find_unused_parameters=True
            )
    else:
        # model = torch.nn.DataParallel(model)
        pass
    # print('Model Done.')

    if is_training:
        # optimizer:
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)
        if args.decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.decay == 'multisteps':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)
        # print('Optimizer Done.')

        return model, optimizer, scheduler, num_outputs
    else:
        return model, num_outputs


def build_prior(args, model, img_num_per_cls, num_classes, num_outputs, device):
    img_num_per_cls = torch.from_numpy(img_num_per_cls).to(device)
    if args.logit_adjust > 0:
        adjustments = img_num_per_cls / img_num_per_cls.sum()
        adjustments = args.logit_adjust * torch.log(adjustments + 1e-12)[None, :]
        if args.ood_metric in ['bkg_c'] and adjustments.shape[1] != num_outputs:
            placeholder = torch.zeros_like(adjustments[:, :num_outputs - num_classes])
            adjustments = torch.cat((adjustments, placeholder), dim=1)
    else:
        if args.ood_metric in ['bkg_c']:
            adjustments = torch.zeros((1, num_outputs), device=device)
        else:
            adjustments = torch.zeros((1, num_classes), device=device)

    return adjustments
