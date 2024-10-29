# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse, os, datetime, time
import os.path as osp
from sklearn.metrics import f1_score
import shutil
from datetime import datetime
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.utils import create_dir, set_random_seed, AverageMeter, save_curve, is_parallel, de_parallel
from utils.ltr_metrics import shot_acc
from utils.common import build_dataset, build_model, build_prior
from models.feat_pool import IDFeatPool

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PASCL for OOD detection in long-tailed recognition')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--num_workers', '--cpus', type=int, default=16, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='./data', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'waterbird'])
    parser.add_argument('--ood_aux_dataset', '--ood_ds', default='TinyImages', choices=['TinyImages', 'VOS', 'NPOS', 'CIFAR', 'Texture'])
    parser.add_argument('--id_class_number', type=int, default=1000, help='for ImageNet subset')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--save_epochs', type=int, default=-1, help='number of epochs to save')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay_epochs', '--de', default=[60,80], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--Lambda', default=0.5, type=float, help='OE loss term tradeoff hyper-parameter')
    parser.add_argument('--Lambda2', default=0.1, type=float, help='Contrastive loss term tradeoff hyper-parameter')
    parser.add_argument('--T', default=0.07, type=float, help='Temperature in NT-Xent loss (contrastive loss)')
    parser.add_argument('--k', default=0.4, type=float, help='bottom-k classes are taken as tail class')
    parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
    # opmitization params:
    parser.add_argument('--logit_adjust', '--tau', default=0., type=float)
    parser.add_argument('--ood_metric', default='oe', choices=['oe', 'bkg_c', 'energy', 'bin_disc', 'mc_disc', 'maha',
                                                               'ada_bin_disc', 'ada_oe', 'ada_energy', 'ada_pascl', 'ada_maha'], help='OOD training metric')
    parser.add_argument('--aux_ood_loss', default='none', choices=['none', 'pascl', 'simclr'], help='Auxilliary (e.g., feature-level) OOD training loss')
    parser.add_argument('--early-stop', action='store_true', default=True, dest='early_stop', help='If true, early stop when lambda dose not change')
    parser.add_argument('--no-early-stop', action='store_false', dest='early_stop')
    parser.add_argument('--w_beta', default=1.0, type=float)
    parser.add_argument('--t_beta', default=2.0, type=float)
    # 
    parser.add_argument('--timestamp', action='store_true', help='If true, attack time stamp after exp str')
    parser.add_argument('--resume', type=str, default='', help='Resume from pre-trained models')
    parser.add_argument('--save_root_path', '--srp', default='./runs', help='data root path')
    # ddp 
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()

    assert args.k>0, "When args.k==0, it is just the OE baseline."

    if args.dataset == 'imagenet':
        # adjust learning rate:
        args.lr *= args.batch_size / 256. # linearly scaled to batch size

    return args


def create_save_path(args, _mkdir=True):
    # mkdirs:
    decay_str = args.decay
    if args.decay == 'multisteps':
        decay_str += '-'.join(map(str, args.decay_epochs)) 
    opt_str = args.opt 
    if args.opt == 'sgd':
        opt_str += '-m%s' % args.momentum
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s' % (args.epochs, args.batch_size, opt_str, args.lr, args.wd, decay_str)
    reweighting_fn_str = 'sign' 
    loss_str = '%s-Lambda%s-Lambda2%s-T%s-%s' % \
                (args.ood_metric + '-' + args.aux_prior_type + '-' + args.aux_ood_loss, 
                 args.Lambda, args.Lambda2, args.T, reweighting_fn_str)
    if args.imbalance_ratio < 1:
        if args.logit_adjust > 0:
            lt_method = 'LA%s' % args.logit_adjust
        else:
            lt_method = 'none'
        loss_str = lt_method + '-' + loss_str
    loss_str += '-k%s'% (args.k)
    exp_str = '%s_%s' % (opt_str, loss_str)
    if args.timestamp:
        exp_str += '_%s' % datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s%d-lt' % (args.dataset, args.id_class_number)
    save_dir = osp.join(args.save_root_path, dataset_str, args.model, exp_str)
    if _mkdir:
        create_dir(save_dir)
        print('Saving to %s' % save_dir)

    return save_dir


def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(gpu_id, ngpus_per_node, args): 

    save_dir = args.save_dir

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'

    synthesis_ood_flag = args.ood_aux_dataset in ['VOS', 'NPOS']
    require_feats_flag = 'maha' in args.ood_metric
    num_classes, train_loader, test_loader, ood_loader, train_sampler, img_num_per_cls_and_ood = build_dataset(args, ngpus_per_node)
    img_num_per_cls = img_num_per_cls_and_ood[:num_classes]

    model, optimizer, scheduler, num_outputs = build_model(args, num_classes, device, gpu_id)
    if require_feats_flag:
        model.id_feat_pool = IDFeatPool(num_classes, sample_num=max(img_num_per_cls), 
                                        feat_dim=model.penultimate_layer_dim, device=device)

    adjustments = build_prior(args, model, img_num_per_cls, num_classes, num_outputs, device)

    # train:
    if args.resume:
        # ckpt = torch.load(osp.join(save_dir, 'latest.pth'), map_location='cpu')
        ckpt = torch.load(osp.join(args.resume, 'latest.pth'), map_location='cpu')
        if is_parallel(model):
            ckpt['model'] = {'module.' + k: v for k, v in ckpt['model'].items()}
        model.load_state_dict(ckpt['model'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        except:
            pass 
        start_epoch = ckpt['epoch']+1 
        best_overall_acc = ckpt['best_overall_acc']
        training_losses = ckpt['training_losses']
        test_clean_losses = ckpt['test_clean_losses']
        f1s = ckpt['f1s']
        overall_accs = ckpt['overall_accs']
        many_accs = ckpt['many_accs']
        median_accs = ckpt['median_accs']
        low_accs = ckpt['low_accs']
    else:
        training_losses, test_clean_losses = [], []
        f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
        best_overall_acc = 0
        start_epoch = 0
    # print('Resume Done.')

    fp = open(osp.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(osp.join(save_dir, 'val_log.txt'), 'a+')
    shutil.copyfile('models/base.py', f'{save_dir}/base.py')
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        start_time = time.time()

        model.train()
        training_loss_meter = AverageMeter()
        current_lr = scheduler.get_last_lr()
        pbar = zip(train_loader, ood_loader)
        # if args.ddp and rank == 0:
        #     pbar = tqdm(pbar, desc=f'Epoch: {epoch:03d}/{args.epochs:03d}', total=len(train_loader))
        stop_flag = False
        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(pbar):
            in_data = torch.cat([in_data[0], in_data[1]], dim=0) # shape=(2*N,C,H,W). Two views of each image.
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data = ood_data.to(device)

            # forward:
            if not synthesis_ood_flag and not require_feats_flag:
                all_data = torch.cat([in_data, ood_data], dim=0) # shape=(2*Nin+Nout,C,W,H)
                in_loss, ood_loss, aux_loss = model(all_data, mode='calc_loss', labels=labels, adjustments=adjustments, args=args)
            elif synthesis_ood_flag:
                in_loss, ood_loss, aux_loss, id_feats = \
                    model(in_data, mode='calc_loss', labels=labels, adjustments=adjustments, args=args, ood_data=ood_data, return_features=True)
                ood_loader.update(id_feats.detach().clone(), labels)
            elif require_feats_flag:
                all_data = torch.cat([in_data, ood_data], dim=0) # shape=(2*Nin+Nout,C,W,H)
                num_ood = len(ood_data)
                in_loss, ood_loss, aux_loss, id_feats = \
                    model(all_data, mode='calc_loss', labels=labels, adjustments=adjustments, args=args, return_features=True)

            loss: torch.Tensor = in_loss + args.Lambda * ood_loss + args.Lambda2 * aux_loss
            if torch.isnan(loss):
                print('Warning: Loss is NaN. Training stopped.')
                stop_flag = True
                break
            if require_feats_flag:
                model.id_feat_pool.update(id_feats[-num_ood:].detach().clone(), torch.cat((labels, labels)))

            # backward:
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            if rank == 0 and batch_idx % 100 == 0:
                train_str = '%s epoch %d batch %d (train): loss %.4f (%.4f, %.4f, %.4f) | lr %s' % (
                    datetime.now().strftime("%D %H:%M:%S"),
                    epoch, batch_idx, loss.item(), in_loss.item(), ood_loss.item(), aux_loss.item(), current_lr) 
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        if stop_flag:
            print('Use the model at epoch', epoch - 1)
            break

        # lr update:
        scheduler.step()

        if rank == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            preds_list, labels_list = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits, features = model(data, return_features=True)
                    in_logits = de_parallel(model).parse_logits(logits, features, args.ood_metric, logits.shape[0])[0]
                    pred = in_logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(in_logits, labels)
                    test_acc_meter.append((in_logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
                    preds_list.append(pred)
                    labels_list.append(labels)

            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

            overall_acc = (preds == labels).sum().item() / len(labels)
            f1 = f1_score(labels, preds, average='macro')

            many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)

            test_clean_losses.append(test_loss_meter.avg)
            f1s.append(f1)
            overall_accs.append(overall_acc)
            many_accs.append(many_acc)
            median_accs.append(median_acc)
            low_accs.append(low_acc)

            val_str = '%s epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | time %s' % \
                        (datetime.now().strftime("%D %H:%M:%S"), epoch, overall_acc, many_acc, median_acc, low_acc, f1, time.time()-start_time) 
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

            # save curves:
            training_losses.append(training_loss_meter.avg)
            save_curve(args, save_dir, training_losses, test_clean_losses, 
                       overall_accs, many_accs, median_accs, low_accs, f1s)

            # save best model:
            model_state_dict = de_parallel(model).state_dict()
            if overall_accs[-1] > best_overall_acc and epoch >= args.epochs * 0.75:
                best_overall_acc = overall_accs[-1]
                torch.save(model_state_dict, osp.join(save_dir, 'best_clean_acc.pth'))
            
            # save feature pool
            if synthesis_ood_flag:
                ood_loader.save(osp.join(save_dir, 'id_feats.pth'))
            elif require_feats_flag:
                model.id_feat_pool.save(osp.join(save_dir, 'id_feats.pth'))  # exactly the same

            # save pth:
            torch.save({
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 
                'best_overall_acc': best_overall_acc,
                'training_losses': training_losses, 
                'test_clean_losses': test_clean_losses, 
                'f1s': f1s, 
                'overall_accs': overall_accs, 
                'many_accs': many_accs, 
                'median_accs': median_accs, 
                'low_accs': low_accs, 
                }, 
                osp.join(save_dir, 'latest.pth'))
            if args.save_epochs > 0 and epoch % args.save_epochs == 0:
                torch.save({
                    'model': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    }, osp.join(save_dir, f'epoch{epoch}.pth'))
                if synthesis_ood_flag:
                    ood_loader.save(osp.join(save_dir, f'id_feats_epoch{epoch}.pth'))
                elif require_feats_flag:
                    model.id_feat_pool.save(osp.join(save_dir, f'id_feats_epoch{epoch}.pth'))  # exactly the same
    
    # Clean up ddp:
    if args.ddp:
        cleanup()

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path(args)
    args.save_dir = save_dir
    with open(f'{save_dir}/args.yaml', 'w+') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    # set random seed, default None
    set_random_seed(args.seed)

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)