import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import cv2, numpy, math
import lib.train.data.transforms as tfm
import os, random

from lib.train.data import opencv_loader
from lib.utils.box_ops import return_iou_boxes, box_xywh_to_xyxy, box_cxcywh_to_xyxy


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, 
                 search_transform=transforms.ToTensor(), joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  search_transform,
                          'grounding':  transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 center_sampling=False, mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.center_sampling = center_sampling


    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
            data['grounding_images'], data['grounding_anno'], data['grounding_masks'] = self.transform['joint'](
                image=data['grounding_images'], bbox=data['grounding_anno'], mask=data['grounding_masks'], new_roll=False)

        # Unsupervised Tracking: save original full search frame
        data['search_frames_path'] = tuple(data['search_images'])
        data['search_ori_anno'] = data['search_anno']
        
        for s in ['template', 'search', 'grounding']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            if s in ['template', 'grounding']:
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                                self.output_sz[s], masks=data[s + '_masks'])
            else:
                # For Full search image
                crops, boxes, att_mask, mask_crops, search_range = prutils.resize_search_img(data[s + '_images'], jittered_anno,
                                                                    data[s + '_anno'], self.search_area_factor[s],
                                                                    self.output_sz[s], masks=data[s + '_masks'])
                # Unsupervised Tracking: the left top coord of the resized image in the padding image
                data['search_range'] = search_range
            
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data



class TemplateProcessing(BaseProcessing):
    def __init__(self, settings, *args, **kwargs):
        """Crops templates by the predicted boxes of grounding process from original images
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = settings.search_area_factor
        self.output_sz = settings.output_sz
        self.center_jitter_factor = settings.center_jitter_factor
        self.scale_jitter_factor = settings.scale_jitter_factor
        self.settings = settings
        self.local_rank = settings.local_rank if settings.local_rank > -1 else 0
        # self.label_function_params = label_function_params
        self.template_sz = self.output_sz['template']
        self.transform['joint'] = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                                tfm.RandomHorizontalFlip(probability=0.5))

        self.transform['template'] = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                                   tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                                   tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _get_templates_and_anno(self, search_path, pred_dict, search_range, 
                                search_anno=None, search_ori_anno=None):
        """
        args:
            grounding_path - The path of the grounding_path
            grounding_dict - The predicted boxes of grounding process
            grounding_image_size - The top-left coords of resized grounding images
        returns:
            frames - The origin images of grounding patch, without any resize or transform
            scale_boxes - The target boxes to the size of the original images which are transformed from the predicted
            boxes
        """
        if isinstance(search_path[0], str):
            frames = [opencv_loader(path) for path in search_path]
        else:
            frames = search_path
        
        # get each original image shape the tensor shape is [b* 3 (h,w,c)]
        original_shapes = torch.tensor([img.shape for img in frames], device=self.local_rank)
        # the predicted boxes
        pred_boxes = torch.round(pred_dict * self.output_sz['search'])
        # Compute the IOU boxes between the predicted boxes and the resized image in the grounding image
        iou_boxes = return_iou_boxes(box_xywh_to_xyxy(search_range), box_cxcywh_to_xyxy(pred_boxes))  # (x1, y1, w, h)
        # Compute the iou boxes' relative position in the resize image
        # Compute x y relative position
        iou_boxes[:, 0:2] = torch.sub(iou_boxes[:, 0:2], search_range[:, 0:2]).clamp(min=0)
        scale_factor = torch.div(original_shapes[:, 1], search_range[:, 2]).unsqueeze(-1)
        # the size of predict boxes in original image
        scale_boxes = iou_boxes * scale_factor
        
        return frames, scale_boxes

    def __call__(self, data: TensorDict, search_path, pred_dict, search_range, search_anno=None, search_ori_anno=None):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
            grounding_path - The path of the grounding_path
            grounding_dict - The predicted boxes of grounding process
            grounding_image_coords - The top-left coords of resized grounding images
        returns:
            TensorDict - output data block with following additional fields:
                'template_images', 'template_att',
        """
        frames, template_boxes = self._get_templates_and_anno(search_path, pred_dict, search_range, 
                                                              search_anno=search_anno, search_ori_anno=search_ori_anno)
        memory, annos, atts, masks = [], [], [], []
        s = 'template'
        for i, img in enumerate(frames):
            H, W, _ = img.shape
            template_mask = torch.zeros((H, W))
            # Apply joint transforms
            if self.transform['joint'] is not None:
                template_images, template_anno, template_masks = self.transform['joint'](
                                    image=[img], bbox=[template_boxes[i]], mask=[template_mask])

            jittered_anno = [self._get_jittered_box(a, s) for a in template_anno]

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.template_center_crop(template_images, jittered_anno,
                                                                              template_anno,
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=template_masks)
            # Apply transforms
            template_images, template_anno, template_att, template_mask = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            
            memory.append(template_images[0].to(self.local_rank))
            annos.append(template_anno[0].to(self.local_rank))
            # atts.extend(template_att)
            # masks.extend(template_mask)

        memory = torch.stack(memory, dim=0)
        annos = torch.stack(annos, dim=0)
        # atts = torch.stack(atts, dim=0)
        # masks = torch.stack(masks, dim=0)
        
        if not 'memory_images' in data.keys():
            data['memory_images'] = [memory]
            data['memory_annos'] = [annos]
        else:
            data['memory_images'].append(memory)
            data['memory_annos'].append(annos)
        # data['memory_att'] = atts
        # data['memory_masks'] = masks
        return data
