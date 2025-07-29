"""
Basic SSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sstrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.sstrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.sstrack.vit_dropmae import vit_base_dropmae_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.train.data.processing import TemplateProcessing
from lib.utils.ce_utils import generate_mask_cond, generate_attn_box_mask

from lib.utils.nt_xent import NTXentLoss


def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features
    """
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        # mask = mask.clamp(min=1e-6, max=1-1e-6)
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate
    
    
class SSTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", cfg=None, settings=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.track_query = None
        
        self.cfg = cfg
        self.processing = TemplateProcessing(settings) if settings is not None else None
        
        # contrastive loss and KL loss
        self.contrastive_loss = cfg.TRAIN.CONTRASTIVE_LOSS
        self.device = self._get_device()
        if cfg.TRAIN.CONTRASTIVE_LOSS:
            self.NTXentLoss = NTXentLoss(device=self.device, batch_size=settings.batchsize,
                                         temperature=0.5, use_cosine_similarity=True)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("\nRunning on:", device)

        # if device == 'cuda':
        #     device_name = torch.cuda.get_device_name()
            # print("The device name is:", device_name)
            # cap = torch.cuda.get_device_capability(device=None)
            # print("The capability of this device is:", cap, '\n')
        return device

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                grounding=None,
                grounding_masks=None,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None,
                training=False
                ):
        if training:
            return self.train_forward(template=template, search=search, 
                                      grounding=grounding, grounding_masks=grounding_masks,
                                      attn_box_mask_z=attn_box_mask_z,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate, 
                                    return_last_attn=return_last_attn,
                                    data=data)
        else:
            return self.inference(template, search, 
                                  attn_box_mask_z=attn_box_mask_z,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate, 
                            return_last_attn=return_last_attn,
                            data=data)
    
    def train_forward(self, template: torch.Tensor,
                search: torch.Tensor,
                grounding: torch.Tensor,
                grounding_masks=None,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None
                ):
        assert isinstance(search, list), "The type of search is not List"
        bs = search[0].size(0)
        out_dict = []
        view_feats = []
        
        # ----------- Forward Tracking -----------
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        track_query=self.track_query,
                                        # attn_box_mask_z=attn_box_mask_z
                                        )
            enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
            
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = enc_opt * att
            
            out = self.forward_head(opt, None)
            out_dict.append(out)
            
            if self.contrastive_loss:  # For contrastive loss
                view_feats.append(enc_opt)
                
            # Crop search frame according to predict results
            search_range = data['search_range'][i]  # (T, B, 4)
            pred_boxes = out['pred_boxes'].view(-1, 4).detach()
            search_frames_list = []
            for b in range(bs):
                search_frames_list.append(data['search_frames_path'][b][i])
            self.processing(data, search_frames_list, pred_boxes, search_range, 
                            search_ori_anno=data['search_ori_anno'][i], search_anno=data['search_anno'][i])
        
        
        # ----------- Backward Tracking -----------
        if 'memory_images' in data.keys():
            memory_list = data['memory_images']
            memory_annos = torch.stack(data['memory_annos'], dim=0)
            
            memory_box_mask = []  # CE module
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                for i in range(len(memory_list)):
                    memory_box_mask.append(generate_mask_cond(self.cfg, memory_list[0].shape[0], memory_list[i].device, memory_annos[i]))
                memory_box_mask = torch.cat(memory_box_mask, dim=1)

            for i in range(len(grounding)):
                x, aux_dict = self.backbone(z=memory_list, x=grounding[i],
                                            ce_template_mask=memory_box_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, 
                                            track_query=self.track_query,
                                            )
                enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

                if self.backbone.add_cls_token:
                    self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
                    
                att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
                opt = enc_opt * att

                out = self.forward_head(opt, None)
                out_dict.append(out)
                
                if self.contrastive_loss:  # For contrastive loss
                    view_feats.append(enc_opt)
        
        # ------- Contrastive Loss during the Backward Tracking -------
        contrastive_loss = torch.tensor(0., dtype=torch.float).cuda()
        if self.contrastive_loss:
            box_mask_x = []
            for i in range(len(grounding)):
                box_mask_x.append(generate_attn_box_mask(self.cfg, grounding[i].shape[0], grounding[i].device,
                                        out_dict[-len(grounding)+i]['pred_boxes'][:, 0], box_type='search_feat'))
            # filter all 0 box_masks: only two augmented views
            sampled_boxmask_x, sampled_view_feats = [[] for _ in range(len(grounding))], [[] for _ in range(len(grounding))]
            for i in range(bs):
                if box_mask_x[0][i].sum() == 0 or box_mask_x[1][i].sum() == 0:  # skip sample
                    continue
                for j in range(len(grounding)):
                    sampled_boxmask_x[j].append(box_mask_x[j][i])
                    sampled_view_feats[j].append(view_feats[-len(grounding)+j][i])
            
            if len(sampled_boxmask_x[0]) > 0:
                grounding_view_feats = []
                for j in range(len(grounding)):
                    boxmask = torch.cat(sampled_boxmask_x[j], dim=0).flatten(1, 2)
                    view_feats = torch.stack(sampled_view_feats[j], dim=0)
                    grounding_view_feats.append(agg_lang_feat(view_feats, boxmask, pool_type="average")) # BNC -> BC
                contrastive_loss = contrastive_loss + self.NTXentLoss(grounding_view_feats[0], grounding_view_feats[1])
        
            return out_dict, contrastive_loss
        
        return out_dict, None


    def inference(self, template: torch.Tensor,
                search: torch.Tensor,
                attn_box_mask_z=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                data=None
                ):
        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template, x=search[i],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        track_query=self.track_query,
                                        )
            enc_opt = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :1].clone()).detach() # stop grad  (B, N, C)
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = enc_opt * att
            
            out = self.forward_head(opt, None)
            out_dict.append(out)
        
        return out_dict
    
        
    def forward_head(self, enc_opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_sstrack(cfg, training=True, settings=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_dropmae_ce':
        backbone = vit_base_dropmae_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        )
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cfg=cfg,
        settings=settings,
    )

    return model
