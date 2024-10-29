# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from typing import Tuple, Optional

from utils.loss_fn import my_cl_loss_fn3, stable_imbce
from models.feat_pool import IDFeatPool


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.return_features = False

        self.aux_linear = None
        self.projection = None
        self.lambda_linear = None

        self.id_feat_pool: Optional["IDFeatPool"] = None

        self.num_classes = 1
        self.penultimate_layer_dim = 1

    def build_aux_layers(self):
        self.aux_linear = nn.Linear(1, 1)
        self.projection = nn.Sequential(
            nn.Linear(self.penultimate_layer_dim, self.penultimate_layer_dim), 
            nn.ReLU(), 
            nn.Linear(self.penultimate_layer_dim, 128)
        )

        self.lambda_linear = nn.Linear(self.penultimate_layer_dim, self.num_classes)
        self.lambda_linear.bias.data.fill_(0.0)
    
    def forward_features(self, x):
        raise NotImplementedError
    
    def forward_classifier(self, p4):
        raise NotImplementedError

    def forward_aux_classifier(self, x):
        return self.aux_linear(x) # (11)
    
    def forward_lambda(self, x, _prob=None, eps=1e-4) -> torch.Tensor:
        lambd = self.lambda_linear(x).exp()
        # _min, _max = eps/(_prob+eps), (1+eps)/(_prob+eps)
        # lambd = torch.minimum(torch.maximum(lambd, _min), _max)
        return lambd.squeeze()

    def forward_projection(self, p4):
        projected_f = self.projection(p4) # (10)
        projected_f = F.normalize(projected_f, dim=1)
        return projected_f

    def forward(self, x, mode='forward_only', **kwargs):
        p4 = self.forward_features(x)
        logits = self.forward_classifier(p4)

        ood_p4 = kwargs.pop('ood_data', None)
        if ood_p4 is not None:
            ood_logits = self.forward_classifier(ood_p4)
            logits = torch.cat((logits, ood_logits), dim=0)
            p4 = torch.cat((p4, ood_p4), dim=0)

        return_features = kwargs.pop('return_features', False) or self.return_features
        if mode == 'forward_only':
            return (logits, p4) if return_features else logits
        elif mode == 'calc_loss':
            res = self.calc_loss(logits, p4, **kwargs)
            ret_p4 = p4 if ood_p4 is None else p4[:-len(ood_p4)]
            return (*res, ret_p4) if return_features else res
        else:
            raise NotImplementedError(mode)
    
    def calc_loss(self, logits, p4, labels, adjustments, args, use_imood=True):
        in_labels = torch.cat([labels, labels], dim=0)
        num_sample, total_num_in = logits.shape[0], in_labels.shape[0]
        assert num_sample > total_num_in
        device = in_labels.device
        metric = args.ood_metric

        in_sample_in_logits, in_sample_ood_logits, ood_sample_in_logits, ood_sample_ood_logits \
            = self.parse_logits(logits, p4, metric, total_num_in)

        in_loss, ood_loss, aux_ood_loss = \
            torch.zeros((1,), device=device), torch.zeros((1,), device=device), torch.zeros((1,), device=device)

        if not metric.startswith('ada_'):

            in_loss += F.cross_entropy(in_sample_in_logits + adjustments, in_labels)

            if metric == 'oe':
                ood_loss += -(ood_sample_ood_logits.mean(1) - torch.logsumexp(ood_sample_ood_logits, dim=1)).mean()
            elif metric == 'energy':
                Ec_out = -torch.logsumexp(ood_sample_ood_logits, dim=1)
                Ec_in = -torch.logsumexp(in_sample_ood_logits, dim=1)
                m_in, m_out = -23 if self.num_classes == 10 else -27, -5  # cifar10/100
                # 0.2 * 0.5 = 0.1, the default loss scale in official Energy OOD
                ood_loss += (torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean()) * 0.2
            elif metric == 'bkg_c':
                ood_labels = torch.full_like(in_labels[:1], self.num_classes)
                ood_loss += F.cross_entropy(ood_sample_ood_logits, ood_labels)
            elif metric == 'bin_disc':
                ood_labels = torch.zeros((num_sample,), device=device)
                ood_labels[:total_num_in] = 1.
                ood_logits = torch.cat((in_sample_ood_logits, ood_sample_ood_logits), dim=0).squeeze(1)
                ood_loss += F.binary_cross_entropy_with_logits(ood_logits, ood_labels)
            elif metric == 'mc_disc':
                ood_labels = torch.zeros((num_sample,), device=device)
                ood_labels[:total_num_in] = 1.  # id: cls0; ood: cls1
                ood_logits = torch.cat((in_sample_ood_logits, ood_sample_ood_logits), dim=0)
                ood_loss += F.cross_entropy(ood_logits, ood_labels)
            else:
                raise NotImplementedError(metric)

        else:
            ood_logits = torch.cat((in_sample_ood_logits, ood_sample_ood_logits), dim=0)
            ood_logits = self.parse_ada_ood_logits(ood_logits, metric)

            ood_labels = torch.zeros((num_sample,), device=device)
            ood_labels[:total_num_in] = 1.

            cls_prior = F.softmax(adjustments, dim=1)

            min_thresh = 1e-4
            lambd = self.forward_lambda(p4).squeeze().clamp(min=min_thresh)

            smoothing = 0.2
            m_in_labels: torch.Tensor = F.one_hot(in_labels, num_classes=self.num_classes)
            in_posterior = m_in_labels * (1 - smoothing) + smoothing / self.num_classes
            ood_posterior = F.softmax(ood_sample_in_logits.detach(), dim=1)
            cls_posterior = torch.cat((in_posterior, ood_posterior))
            beta = (lambd * cls_posterior / cls_prior).mean(dim=1) #.clamp(min=1e-1, max=1e+1)
            ood_loss += (beta.log() + ood_logits.detach().sigmoid().log()).relu().mean()
            
            beta = beta.detach()
            delta = (beta + (beta - 1.) * torch.exp(ood_logits.detach())).clamp(min=1e-1, max=1e+1)
            delta = torch.cat((delta[:total_num_in].clamp(min=1.), 
                                delta[total_num_in:].clamp(max=1.)), dim=0)
            ood_logits = ood_logits - delta.log()

            ood_loss += F.binary_cross_entropy_with_logits(ood_logits, ood_labels)

            if metric == 'ada_oe':  # add original OE loss 
                ood_loss += -(ood_sample_ood_logits.mean(1) - torch.logsumexp(ood_sample_ood_logits, dim=1)).mean()

            in_sample_in_logits = in_sample_in_logits + adjustments
            in_loss += F.cross_entropy(in_sample_in_logits, in_labels)
        
        aux_ood_loss += self.calc_aux_loss(p4, labels, args)

        return in_loss, ood_loss, aux_ood_loss

    def calc_aux_loss(self, p4, labels, args):
        device = p4.device
        aux_loss = torch.zeros(1, device=device)
        num_in = labels.shape[0]

        if 'pascl' == args.aux_ood_loss:
            if not hasattr(self, 'cl_loss_weights'):
                _sigmoid_x = torch.linspace(-1, 1, self.num_classes).to(device)
                _d = -2 * args.k + 1 - 0.001 # - 0.001 to make _d<-1 when k=1
                self.register_buffer('cl_loss_weights', torch.sign((_sigmoid_x-_d)))

            tail_idx = labels >= round((1-args.k)*self.num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            all_f = self.forward_projection(p4)
            f_id_view1, f_id_view2 = all_f[0:num_in], all_f[num_in:2*num_in]
            f_id_tail_view1 = f_id_view1[tail_idx] # i.e., 6,7,8,9 in cifar10
            f_id_tail_view2 = f_id_view2[tail_idx] # i.e., 6,7,8,9 in cifar10
            labels_tail = labels[tail_idx]
            f_ood = all_f[2*num_in:]
            if torch.sum(tail_idx) > 0:
                aux_loss += my_cl_loss_fn3(
                    torch.stack((f_id_tail_view1, f_id_tail_view2), dim=1), f_ood, labels_tail, temperature=args.T,
                    reweighting=True, w_list=self.cl_loss_weights
                )
        elif 'simclr' == args.aux_ood_loss:
            ood_logits = self.projection(p4)[:, 0]
            ood_labels = torch.zeros((len(p4),), device=device)
            assert len(p4) > num_in*2
            ood_labels[:num_in*2] = 1.
            aux_loss = F.binary_cross_entropy_with_logits(ood_logits, ood_labels)
        
        elif 'cocl' == args.aux_ood_loss:
            from utils.loss_fn import compute_dist

            Lambda2, Lambda3 = 0.05, 0.1
            temperature, margin = 0.07, 1.0
            headrate, tailrate = 0.4, 0.4

            f_id_view = p4[:2*num_in]
            f_ood = p4[2*num_in:]
            head_idx = labels<= round(headrate*self.num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            tail_idx = labels>= round((1-tailrate)*self.num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            f_id_head_view = f_id_view[head_idx] # i.e., 6,7,8,9 in cifar10
            f_id_tail_view = f_id_view[tail_idx] # i.e., 6,7,8,9 in cifar10
            labels_tail = labels[tail_idx]

            # OOD-aware tail class prototype learning
            if len(f_id_tail_view) > 0 and Lambda2 > 0:
                ## TODO
                raise NotImplementedError
                logits = self.forward_weight(f_id_tail_view, f_ood, temperature=temperature)
                tail_loss = F.cross_entropy(logits, labels_tail-round((1-tailrate)*self.num_classes))
            else:
                tail_loss = torch.zeros((1, ), device=device)

            # debiased head class learning
            if Lambda3 > 0:
                dist1 = compute_dist(f_ood, f_ood)
                _, dist_max1 = torch.max(dist1, 1)
                positive = f_ood[dist_max1]

                dist2 = torch.randint(low = 0, high= len(f_id_head_view), size = (1, len(f_ood))).to(device).squeeze()
                negative = f_id_head_view[dist2]
                
                triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
                head_loss = triplet_loss(f_ood, positive, negative)
            else:
                head_loss = torch.zeros((1, ), device=device)
            
            aux_loss = tail_loss * Lambda2 + head_loss * Lambda3
        
        return aux_loss

    def parse_logits(self, all_logits, all_features, metric, num_in) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(x in metric for x in ['bin_disc']):
            in_sample_in_logits = all_logits[:num_in, :-1]
            in_sample_ood_logits = all_logits[:num_in, -1:]
            ood_sample_in_logits = all_logits[num_in:, :-1]
            ood_sample_ood_logits = all_logits[num_in:, -1:]
        elif any(x in metric for x in ['mc_disc']):
            in_sample_in_logits = all_logits[:num_in, :-2]
            in_sample_ood_logits = all_logits[:num_in, -2:]
            ood_sample_in_logits = all_logits[num_in:, :-2]
            ood_sample_ood_logits = all_logits[num_in:, -2:]
        elif any(x in metric for x in ['msp', 'oe', 'bkg_c', 'energy']):  
            in_sample_in_logits = all_logits[:num_in, :]
            in_sample_ood_logits = all_logits[:num_in, :]
            ood_sample_in_logits = all_logits[num_in:, :]
            ood_sample_ood_logits = all_logits[num_in:, :]
        elif any(x in metric for x in ['gradnorm']):
            in_sample_in_logits = all_logits[:num_in, :]
            in_sample_ood_logits = all_features[:num_in, :]
            ood_sample_in_logits = all_logits[num_in:, :]
            ood_sample_ood_logits = all_features[num_in:, :]
        elif any(x in metric for x in ['maha']):
            all_maha_scores = self.calc_maha_score(all_features)
            in_sample_in_logits = all_logits[:num_in, :]
            in_sample_ood_logits = all_maha_scores[:num_in, None]
            ood_sample_in_logits = all_logits[num_in:, :]
            ood_sample_ood_logits = all_maha_scores[num_in:, None]
        else:
            raise NotImplementedError('parse_logits %s' % metric)
        
        return in_sample_in_logits, in_sample_ood_logits, ood_sample_in_logits, ood_sample_ood_logits

    def parse_ada_ood_logits(self, ood_logits, metric, project=True):
        if any(x in metric for x in ['bin_disc']):
            pass
        else:
            if any(x in metric for x in ['msp', 'oe']):
                ood_logits = F.softmax(ood_logits, dim=1).max(dim=1, keepdim=True).values - 1. / self.num_classes  # MSP
            elif any(x in metric for x in ['energy']):
                ood_logits = torch.logsumexp(ood_logits, dim=1, keepdim=True)
            elif any(x in metric for x in ['maha']):
                pass  # already calculated
            elif any(x in metric for x in ['gradnorm']):
                ood_logits = [self.calc_gradnorm_per_sample(f) for f in ood_logits]
                ood_logits = torch.tensor(ood_logits).view(-1, 1).cuda()
            else:
                raise NotImplementedError(metric)
            if project:
                ood_logits = self.forward_aux_classifier(ood_logits)

        return ood_logits.squeeze(1)

    def calc_maha_score(self, features):
        assert self.id_feat_pool is not None
        if self.training and not self.id_feat_pool.ready():
            return torch.zeros(len(features), device=features.device)
        return self.id_feat_pool.calc_maha_score(features, force_calc=self.training)
    
    def calc_gradnorm_per_sample(self, features, targets=None, temperature=1.):
        assert len(features.shape) == 1
        self.requires_grad_(True)

        features = features.view(1, -1)
        features = Variable(features.cuda(), requires_grad=True)
        self.zero_grad()
        outputs = self.forward_classifier(features) / temperature

        if targets is None:
            targets = torch.ones((1, self.num_classes)).cuda() / self.num_classes
        
        kl_loss = F.kl_div(outputs.softmax(dim=-1).log(), targets.softmax(dim=-1), reduction='sum')
        kl_loss.backward()
        layer_grad = self.linear.weight.grad.data
        gradnorm = torch.sum(torch.abs(layer_grad))

        self.requires_grad_(False)

        return gradnorm


def get_ood_scores(model: BaseModel, images, metric, adjustments):
    logits, features = model(images, return_features=True)
    in_logits, ood_logits = model.parse_logits(logits, features, metric, logits.shape[0])[:2]

    if metric.startswith('ada_'):
        ood_logits = model.parse_ada_ood_logits(ood_logits, metric, project=False)

        prior = F.softmax(adjustments, dim=1)
        posterior = F.softmax(in_logits, dim=1)
        out_adjust = (posterior / prior).mean(dim=1).log()

        # 1.0 for bin_disc, 0.1 for msp, 1.0 for energy, 0.02 for pascl, 0.01 for maha
        scale_dict = {'ada_msp': 0.1, 'ada_energy': 1.0, 'ada_bin_disc': 1.0, 'ada_maha': 0.01, 'ada_gradnorm': 10}
        ood_logits += out_adjust * scale_dict[metric]
        scores = - ood_logits

    else:
        prior = F.softmax(adjustments, dim=1)
        posterior = F.softmax(in_logits, dim=1)

        if metric == 'msp':
            # The larger MSP, the smaller uncertainty
            scores = - F.softmax(logits, dim=1).max(dim=1).values 
        elif metric == 'energy':
            # The larger energy, the smaller uncertainty
            tau = 1.
            scores = - tau * torch.logsumexp(logits / tau, dim=1)
        elif metric == 'bkg_c':
            # The larger softmax background-class prob, the larger uncertainty
            scores = F.softmax(ood_logits, dim=1)[:, -1]
        elif metric == 'bin_disc':
            # The larger sigmoid prob, the smaller uncertainty
            scores = 1. - ood_logits.squeeze(1).sigmoid()
        elif metric == 'mc_disc':
            # The larger softmax prob, the smaller uncertainty
            scores = F.softmax(ood_logits, dim=1)[:, 1] 
        elif metric == 'rp_msp':
            # The larger MSP, the smaller uncertainty
            scores = - (F.softmax(logits, dim=1) - .01 * F.softmax(adjustments, dim=1)).max(dim=1).values 
        elif metric == 'rp_gradnorm':
            # The larger GradNorm, the smaller uncertainty
            prior = F.softmax(adjustments, dim=1)
            scores = [model.calc_gradnorm_per_sample(feat, targets=prior) for feat in features]
            scores = - torch.tensor(scores)
        elif metric == 'gradnorm':
            # The larger GradNorm, the smaller uncertainty
            scores = [model.calc_gradnorm_per_sample(feat) for feat in features]
            scores = - torch.tensor(scores)
        elif metric == 'rw_energy':
            # The larger energy, the smaller uncertainty
            tau = 1.
            prior = F.softmax(adjustments, dim=1)
            posterior = F.softmax(logits, dim=1)
            rweight = 1. - (posterior * prior).sum(dim=1) / (posterior.norm(2, dim=1) * prior.norm(2, dim=1))
            scores = - tau * torch.logsumexp(logits / tau, dim=1) * rweight
        elif metric == 'maha':
            scores = - ood_logits[:, 0]  # already calculated
        else:
            raise NotImplementedError('OOD inference metric: ', metric)

    return in_logits, scores