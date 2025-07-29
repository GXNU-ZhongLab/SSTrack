from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class SSTrackActor(BaseActor):
    """ Actor for training MMTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, contrastive_loss = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data, contrastive_loss=contrastive_loss)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []
        grounding_list = []
        grounding_masks_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)
        
        for i in range(self.settings.num_grounding):
            grounding_img_i = data['grounding_images'][i].view(-1, *data['grounding_images'].shape[2:])
            grounding_list.append(grounding_img_i)
            grounding_masks_i = data['grounding_masks'][i].view(-1, *data['grounding_masks'].shape[2:])
            grounding_masks_list.append(grounding_masks_i)

        # CE module
        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                                    data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            
        out_dict = self.net(template=template_list,
                            search=search_list,
                            grounding=grounding_list,
                            grounding_masks=grounding_masks_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            data=data,
                            training=True,
                            )
        
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, contrastive_loss=None, return_status=True):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['grounding_anno'], self.cfg.DATA.GROUNDING.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        
        grounding_num = gt_dict['grounding_anno'].size(0)
        grounding_ids = list(range(0, len(pred_dict)))[-grounding_num:]
        search_num = len(pred_dict) - grounding_num
        
        for i in range(len(pred_dict)):
            if i in grounding_ids:  # For grounding frames: calc box loss
                # get GT
                gt_bbox = gt_dict['grounding_anno'][i - search_num]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
                gt_gaussian_maps = gt_gaussian_maps_list[i - search_num].unsqueeze(1)

                # Get pred boxes
                pred_boxes = pred_dict[i]['pred_boxes']
                if torch.isnan(pred_boxes).any():
                    raise ValueError("Network outputs is NAN! Stop Training")
                num_queries = pred_boxes.size(1)
                pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
                # (B,4) --> (B,1,4) --> (B,N,4)
                
                # compute giou and iou
                try:
                    giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                except:
                    giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                loss_dict['giou'] = giou_loss
                
                # compute l1 loss
                l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                loss_dict['l1'] = l1_loss
                
                # compute location loss
                if 'score_map' in pred_dict[i]:
                    location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
                else:
                    location_loss = torch.tensor(0.0, device=l1_loss.device)
                loss_dict['focal'] = location_loss
                    
                # weighted sum
                loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
                total_loss += loss
                
            else:
                # calc IoU of search frames
                with torch.no_grad():
                    # get GT
                    search_gt = gt_dict['search_anno'][i-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
                    search_pred_boxes = pred_dict[i]['pred_boxes']
                    if torch.isnan(search_pred_boxes).any():
                        raise ValueError("Network outputs is NAN! Stop Training")
                    num_queries = search_pred_boxes.size(1)
                    search_pred_boxes_vec = box_cxcywh_to_xyxy(search_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                    search_gt_boxes_vec = box_xywh_to_xyxy(search_gt)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
                    # (B,4) --> (B,1,4) --> (B,N,4)
                    try:
                        iou, _ = box_iou(search_pred_boxes_vec, search_gt_boxes_vec)  # (BN,4) (BN,4)
                    except:
                        iou = torch.tensor(0.0).cuda()
                    
            if return_status:
                # status for log
                status = {}
                mean_iou = iou.detach().mean()
                
                if i in grounding_ids:
                    status = {f"{i}frame_Loss/total": loss.item(),
                            f"{i}frame_Loss/giou": giou_loss.item(),
                            f"{i}frame_Loss/l1": l1_loss.item(),
                            f"{i}frame_Loss/Focal": location_loss.item(),
                            f"{i}frame_IoU": mean_iou.item()}
                else:
                    status = {f"{i}frame_IoU": mean_iou.item()}
                    
                total_status.update(status)

        if not contrastive_loss is None:
            total_status.update({f"Cont_Loss": contrastive_loss.item()})
            total_loss += contrastive_loss
        
        if return_status:
            return total_loss, total_status
        else:
            return total_loss
