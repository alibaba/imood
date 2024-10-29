# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import faiss
from time import time


class IDFeatPool(object):
    def __init__(self, class_num, sample_num=1500, feat_dim=512, mode='NPOS', device='cuda:0'):
        self.class_num = class_num
        self.sample_num = sample_num
        self.feat_dim = feat_dim
        self.device = device
        
        self.class_ptr = torch.zeros((class_num,)).to(device)
        self.queue = torch.zeros((class_num, sample_num, feat_dim)).to(device)

        self.mode = mode
        if mode == 'NPOS':
            # Standard Gaussian distribution
            assert faiss.StandardGpuResources
            res = faiss.StandardGpuResources()
            self.KNN_index = faiss.GpuIndexFlatL2(res, self.feat_dim)
            self.K = sample_num // 3
            self.sample_from = sample_num * 2
            self.select = sample_num // 5
            self.pick_nums = 1
            self.ID_points_num = 10
        elif mode == 'VOS':
            self.sample_from = sample_num * 10
            self.select = sample_num // 5
            self.pick_nums = 10
            self.ID_points_num = 1
        else:
            raise NotImplementedError(mode)
    
    def update(self, features, labels):
        if self.queue.device != features.device:
            # self.queue = self.queue.to(features.device)
            features = features.to(self.device)
        if self.queue.dtype != features.dtype:
            self.queue = self.queue.type_as(features)

        unique_labels = torch.unique(labels)
        unique_indices = (unique_labels.view(-1, 1) == labels.view(1, -1)).int().argmax(dim=1)
        self.queue[unique_labels] = torch.cat((self.queue[unique_labels, 1:, :], features[unique_indices][:, None, :]), 1)
        self.class_ptr[unique_labels] = (self.class_ptr[unique_labels] + 1).clamp(max=self.sample_num)
    
    def ready(self):
        return (self.class_ptr >= self.sample_num).all()

    def save(self, path):
        torch.save(self.queue.cpu(), path)
    
    def load(self, path):
        self.queue = torch.load(path, map_location='cpu')[:, :self.queue.shape[1], :].to(self.queue.device)
        self.class_ptr = (self.queue != 0.).any(dim=-1).sum(dim=1).to(self.class_ptr.device)
    
    def __getitem__(self, index):
        if not self.ready() or False:  #  and (self.class_ptr == 0).any()
            no = self.class_num * self.ID_points_num * self.pick_nums
            return torch.randn((no, self.feat_dim)).to(self.device), torch.full((no, ), -1).to(self.device), 
        
        ood_samples, ood_labels = [], []

        if self.mode == 'VOS':
            ood_samples, ood_labels = [], []
            mean_embed_id = self.queue.mean(dim=1, keepdim=True)  # shape(nc,1,ndim)
            X = (self.queue - mean_embed_id).view(-1, self.feat_dim)  # shape(nc*ns,dim)
            covariance = (X.T @ X) / len(X) * 10. + .1
            # covariance += 0.0001 * torch.eye(len(covariance), device=X.device)
            covariance += 1.1 * torch.eye(len(covariance), device=X.device)

            new_dis = MultivariateNormal(torch.zeros(self.feat_dim).cuda(), covariance_matrix=covariance)
            negative_samples = new_dis.rsample((self.sample_from,)) * 2
            prob_density = new_dis.log_prob(negative_samples)
            cur_samples, index_prob = torch.topk(- prob_density, self.select)
            negative_samples = negative_samples[index_prob]

            for ci, miu in enumerate(mean_embed_id):
                rand_ind = torch.randperm(self.select)[:self.pick_nums]
                ood_samples.append(miu + negative_samples[rand_ind])
                ood_labels.extend([ci] * self.pick_nums)

        elif self.mode == 'NPOS':
            mean_embed_id = self.queue.mean(dim=1, keepdim=True)  # shape(nc,1,ndim)
            X = (self.queue - mean_embed_id).view(-1, self.feat_dim)  # shape(nc*ns,dim)
            covariance = (X.T @ X) / len(X) * 10 + .1
            # covariance += 0.0001 * torch.eye(len(covariance), device=X.device)
            covariance += 1.1 * torch.eye(len(covariance), device=X.device)
            # covariance = torch.eye(self.feat_dim).to(self.queue.device)
            
            self.new_dis = MultivariateNormal(torch.zeros(self.feat_dim).to(self.queue.device), 
                                              covariance)
            
            negative_samples = self.new_dis.rsample((self.sample_from,)).to(self.device) * 2

            ood_samples, ood_labels = generate_outliers(self.queue, input_index=self.KNN_index, negative_samples=negative_samples, 
                                                        ID_points_num=self.ID_points_num, K=self.K, select=self.select, 
                                                        sampling_ratio=1.0, pic_nums=self.pick_nums, depth=self.feat_dim,
                                                        cov_mat=1.)
        
        ood_samples = torch.cat(ood_samples).to(self.device)
        ood_labels = torch.tensor(ood_labels).to(self.device)

        return ood_samples, ood_labels
    
    def calc_maha_score(self, samples: torch.Tensor, force_calc=True):
        # samples: shape(n,ndim)
        ns, nc = samples.shape[0], self.class_num

        sample_num_per_cls = self.class_ptr.view(nc, 1)
        valid_mask = (self.queue != 0).any(dim=-1)  # shape(nc,ns)
        assert (valid_mask.sum(dim=1, keepdim=True) == sample_num_per_cls).all()
        mean_embed_id = self.queue.sum(dim=1) / sample_num_per_cls  # shape(nc,ndim)

        if force_calc or not hasattr(self, 'maha_cov_inv'):
            X = (self.queue - mean_embed_id[:, None, :])[valid_mask]  # shape(x,ndim)
            covariance = (X.T @ X) / len(X)  # shape(ndim,ndim), class-agnostic
            covariance += 0.0001 * torch.eye(len(covariance), device=X.device)
            maha_cov_inv = covariance.inverse()[None, :, :]
            setattr(self, 'maha_cov_inv', maha_cov_inv)
        else:
            maha_cov_inv = getattr(self, 'maha_cov_inv')
        
        samples = samples[:, None, :] - mean_embed_id[None, :, :]  # shape(ns,1,ndim) - shape(1,nc,ndim) = shape(ns,nc,ndim)
        samples = samples.view(ns*nc, self.feat_dim, 1)  # shape(ns*nc,ndim,1)
        maha_dist = torch.bmm(torch.bmm(samples.permute(0,2,1), maha_cov_inv.expand(ns*nc,-1,-1)), samples)  # f^T @ Cov^-1 @ f
        maha_dist = maha_dist.view(ns, nc)
        return - torch.max(-maha_dist, dim=1).values


def KNN_dis_search_decrease(target, index, K=50, select=1,):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    #k_th_output_index = output_index[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    #k_th_index = k_th_output_index[minD_idx]
    return minD_idx, k_th_distance


def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000,depth=342):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    target_new = target.view(length, -1, depth)
    #k_th_output_index = output_index[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    # minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i*length + minD_idx[:,i])
    #return torch.cat(point_list, dim=0)
    return target[torch.cat(point_list)]


def generate_outliers(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, cov_mat=0.1, sampling_ratio=1.0, pic_nums=30, depth=342):
    ncls, nsample, ndim = ID.shape
    length, _ = negative_samples.shape
    normed_data = ID / torch.norm(ID, p=2, dim=-1, keepdim=True)

    distance = torch.cdist(normed_data, normed_data.detach())  # shape(ncls, nsample, nsample)
    k_th_distance = -torch.topk(-distance, K, dim=-1)[0][..., -1]  # k-th nearset (smallest distance), shape(ncls, nsample)
    minD_idx = torch.topk(k_th_distance, select, dim=1)[1]  # top-k largest distance, shape(ncls, select)
    minD_idx = minD_idx[:, np.random.choice(select, int(pic_nums), replace=False)]  #shape(ncls, pic_nums)
    cls_idx = torch.arange(ncls).view(ncls, 1)
    data_point_list = ID[cls_idx.repeat(1, pic_nums).view(-1), minD_idx.view(-1)].view(-1, pic_nums, 1, ndim)

    negative_sample_cov = cov_mat*negative_samples.view(1, 1, length, ndim)
    negative_sample_list = (negative_sample_cov + data_point_list).view(-1, pic_nums*length, ndim)

    normed_ood_feat = F.normalize(negative_sample_list, p=2, dim=-1)  #shape(cls, pic_nums*length, 512)
    distance = torch.cdist(normed_ood_feat, normed_data)  # shape(ncls, pic_nums*length, nsample)
    k_th_distance = -torch.topk(-distance, K, dim=-1)[0][..., -1]  # k-th nearset (smallest distance), shape(ncls, pic_nums*length)
    
    k_distance, minD_idx = torch.topk(k_th_distance, ID_points_num, dim=1)  # top-k largest distance, shape(ncls, ID_points_num)
    OOD_labels = torch.arange(normed_data.size(0)).view(-1, 1).repeat(1, ID_points_num).view(-1)
    OOD_syntheses = negative_sample_list[OOD_labels, minD_idx.view(-1)]    #shape(ncls*ID_points_num, 512)
    
    if OOD_syntheses.shape[0]:
        # concatenate ood_samples outside
        OOD_syntheses = torch.chunk(OOD_syntheses, OOD_syntheses.shape[0])
        OOD_labels = OOD_labels.numpy()

    return OOD_syntheses, OOD_labels
