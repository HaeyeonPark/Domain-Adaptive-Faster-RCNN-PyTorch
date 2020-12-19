# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
from maskrcnn_benchmark.data.build import NUM_TARGET_DOMAINS
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

from .loss import make_da_heads_loss_evaluator

#eun0
NUM_TARGET_DOMAINS = 2
class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, NUM_TARGET_DOMAINS + 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x,out_emb=False):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        if out_emb:
            return img_features,t
        else:
            return img_features



class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, NUM_TARGET_DOMAINS+1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

    def forward(self, img_features, da_ins_feature, da_ins_labels, targets=None, anchors_features=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if self.resnet_backbone:
            da_ins_feature = self.avgpool(da_ins_feature)
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        assert len(img_grl_fea)==1 and (da_ins_labels==1).sum().item()==256
        anc_grl_fea = [self.grl_img(fea) for fea in anchors_features]

        ins_grl_fea = self.grl_ins(da_ins_feature)
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        #eun0
        # [num_domains, num_domains, h, w]

        USE_EMB = True
        da_img_features, img_embs = self.imghead(img_grl_fea,out_emb=USE_EMB)
        _, anc_embs = self.imghead(anc_grl_fea,out_emb=USE_EMB)

        img_embs = img_embs.view(img_embs.size(0),-1)
        anc_embs = anc_embs.view(anc_embs.size(0),-1)

        img_embs = torch.nn.functional.normalize(img_embs,p=2,dim=1)
        anc_embs = torch.nn.functional.normalize(anc_embs,p=2,dim=1)


        #ancs = torch.stack([anc_embs[0],anc_embs[0],anc_embs[1],anc_embs[1],anc_embs[2],anc_embs[2]])
        #pos = torch.stack([img_embs[0],img_embs[0],img_embs[1],img_embs[1],img_embs[2],img_embs[2]])
        #neg = torch.stack([img_embs[1],img_embs[2],img_embs[0],img_embs[2],img_embs[0],img_embs[1]])

        ancs = torch.stack([anc_embs[0],anc_embs[1],anc_embs[2]])
        pos = torch.stack([img_embs[0],img_embs[1],img_embs[2]])
        neg = torch.stack([img_embs[1],img_embs[2],img_embs[0]])

        
        # debug
        #margin=3
        margin = 0.2
        tl = compute_triplet_loss(ancs,pos,neg,margin=margin)


        # [num_domains*num_instance,num_domains]
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        
        da_img_consist_features = [torch.nn.functional.softmax(fea,dim=1)for fea in da_img_consist_features]
        da_ins_consist_features = torch.nn.functional.softmax(da_ins_consist_features,dim=1)
        
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )
            losses = {}
            if USE_EMB:
                losses["img_emb_triplet_loss"] = 0.1 * tl
            else:
                losses["img_triplet_loss"] = 0.1 * tl
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}

def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []

def compute_triplet_loss(ancs, pos, neg, margin=0.2, p=2):

    # [num_domains, h, w]

    d_pos = torch.norm(ancs-pos,p,dim=1)
    d_neg = torch.norm(ancs-neg,p,dim=1)
    loss = nn.ReLU()(d_pos-d_neg+margin)
    reduced_loss = torch.mean(loss)
    return reduced_loss


