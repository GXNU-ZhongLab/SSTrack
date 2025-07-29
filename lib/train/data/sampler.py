import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
from lib.train.data.augmentation_CTP import Augmentation_CTP
from lib.utils.box_ops import Corner
import cv2
import copy

def no_processing(data):
    return data


class SelfSupervised_Sampler(torch.utils.data.Dataset):
    """ 
    If an initial frame and a plurality of history frames are sampled, 
    where the history frames use the initial coordinates as pseudo-labels.
    
    
    Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, num_grounding_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, template_sample_range=0):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.num_grounding_frames = num_grounding_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

        # --------------- self-Supervised Tracking: Data Aug ---------------
        self.template_sample_range = template_sample_range
        self.is_synthesized = 0.5  # If data augmentation is not used, set self.is_synthesized = 0
        
        self.template_aug = Augmentation_CTP(
            0,  # cfg.DATASET.SEARCH.SHIFT,
            0.1,  # cfg.DATASET.SEARCH.SCALE,
            0.0,  # cfg.DATASET.SEARCH.BLUR,
            0.0,  # cfg.DATASET.SEARCH.FLIP,
            0,  # cfg.DATASET.SEARCH.COLOR,
            0,  # cfg.DATASET.SEARCH.OCC,
            0.1,  # cfg.DATASET.SEARCH.SHEAR
            0,  # cfg.DATASET.SEARCH.CROP
            0.1
        )
        self.search_aug = Augmentation_CTP(
            0,  # cfg.DATASET.SEARCH.SHIFT,
            0.25,  # SL_settings['SCALE'],  # cfg.DATASET.SEARCH.SCALE,
            0.18,  # SL_settings['BLUR'],  # cfg.DATASET.SEARCH.BLUR,
            0.3,  # SL_settings['FLIP'],  # cfg.DATASET.SEARCH.FLIP,
            0,                            # cfg.DATASET.SEARCH.COLOR,
            0.1,  # SL_settings['OCC'],  # cfg.DATASET.SEARCH.OCC,
            0.5,  # SL_settings['SHEAR'],  # cfg.DATASET.SEARCH.SHEAR
            0,  # cfg.DATASET.SEARCH.CROP
            0.3  # SL_settings['CJ']
        )
        # --------------- self-Supervised Tracking: Data Aug ---------------
        
    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _sample_template_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None
        
        return valid_ids[:num_ids]  # For single template frame

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            is_aug = np.random.random()
            # grounding_frame_ids = None
            
            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0
                
                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        if is_aug > self.is_synthesized:
                            # take the first K frame as a template
                            base_frame_id = self._sample_template_visible_ids(visible, num_ids=self.num_template_frames,
                                                                min_id=0, max_id=len(visible) - self.num_search_frames)
                            # prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                            #                                           min_id=base_frame_id[0] - self.max_gap - gap_increase,
                            #                                           max_id=base_frame_id[0])
                            # if prev_frame_ids is None:
                            #     gap_increase += 5
                            #     continue
                            # template_frame_ids = base_frame_id + prev_frame_ids
                            template_frame_ids = base_frame_id
                            
                            if (not template_frame_ids is None) and (len(template_frame_ids) > 1):
                                template_frame_ids.sort(reverse=False)  # Sort from small to large
                            
                            search_frame_ids = self._sample_visible_ids(visible, 
                                                                        num_ids=self.num_search_frames,
                                                                        # num_ids=self.num_search_frames - 1, # For Unsupervised Tracking
                                                                        min_id=template_frame_ids[-1] + 1, 
                                                                        max_id=template_frame_ids[-1] + self.max_gap + gap_increase)
                            if (not search_frame_ids is None) and (len(search_frame_ids) > 1):
                                search_frame_ids.sort(reverse=False)  # Sort from small to large
                        else:
                            base_frame_id = self._sample_visible_ids(visible, num_ids=1, 
                                                                 min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                            prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                      min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                      max_id=base_frame_id[0])
                            if prev_frame_ids is None:
                                gap_increase += 5
                                continue
                            template_frame_ids = base_frame_id + prev_frame_ids
                            # template_frame_ids = base_frame_id
                            
                            if (not template_frame_ids is None) and (len(template_frame_ids) > 1):
                                template_frame_ids.sort(reverse=False)  # Sort from small to large
                            
                            search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                                    min_id=template_frame_ids[-1] + 1, 
                                                                    max_id=template_frame_ids[-1] + self.max_gap + gap_increase)
                            if (not search_frame_ids is None) and (len(search_frame_ids) > 1):
                                search_frame_ids.sort(reverse=False)  # Sort from small to large
                                
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
                
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                
                # The first template frame is GT, all other template frames use init GT as a pseudo labels
                if is_aug > self.is_synthesized and self.num_template_frames > 1:
                    for k, v in template_anno.items():
                        template_anno[k] = [v[0]] * self.num_template_frames
                
                # --------------- self-Supervised Tracking: Data Aug ---------------
                if is_aug < self.is_synthesized:  # using the synthesized training samples with an ratio of self.is_synthesized
                    target_frame_id = None
                    while target_frame_id is None:
                        target_seq_id, target_visible, target_seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
                        
                        if is_video_dataset:
                            target_frame_id = self._sample_template_visible_ids(target_visible, num_ids=1, min_id=0, 
                                                                                max_id=len(target_visible) - self.num_search_frames)
                        else:
                            target_frame_id = [1]
                            
                    # selecting a init target
                    target_frame, target_anno, _ = dataset.get_frames(target_seq_id, target_frame_id, target_seq_info_dict)
                    target_img, target_box = target_frame[0], target_anno['bbox'][0].numpy().copy()  # tensor [x, y, w, h]

                    # in case the target is too small
                    min_sz = 10
                    target_box[2], target_box[3] = max(target_box[2], min_sz), max(target_box[3], min_sz)
                    target_box = Corner(int(target_box[0]), int(target_box[1]), int(target_box[0] + target_box[2]),
                                        int(target_box[1] + target_box[3]))

                    target_patch = target_img[target_box.y1:target_box.y2, target_box.x1:target_box.x2, :]

                    for t in range(len(template_frames)):
                        template_frames[t], template_anno['bbox'][t] = self.template_aug(template_frames[t], target_patch)
                    for t in range(len(search_frames)):
                        search_frames[t], search_anno['bbox'][t] = self.search_aug(search_frames[t], target_patch)
                # --------------- self-Supervised Tracking: Data Aug ---------------
                
                # grounding_frames: first template frame
                grounding_frames, meta_obj_grounding = [template_frames[0]], meta_obj_train.copy()
                grounding_anno = {}
                for k, v in template_anno.items():
                    grounding_anno[k] = [v[0]]
                
                if self.num_grounding_frames > 1:
                    grounding_frames = grounding_frames * self.num_grounding_frames
                    grounding_anno['bbox'] = grounding_anno['bbox'] * self.num_grounding_frames
                    if 'mask' in grounding_anno:
                        grounding_anno['mask'] = grounding_anno['mask'] * self.num_grounding_frames
                
                H, W, _ = template_frames[0].shape
                # generate mask matrix
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.ones((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.ones((H, W))] * self.num_search_frames
                grounding_masks = grounding_anno['mask'] if 'mask' in grounding_anno else [torch.ones((H, W))] * self.num_grounding_frames
                
                data = TensorDict({'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'template_masks': template_masks,
                                'search_images': search_frames,  # ori image
                                'search_anno': search_anno['bbox'],
                                'search_masks': search_masks,
                                'grounding_images': grounding_frames,  # ori image
                                'grounding_anno': grounding_anno['bbox'],
                                'grounding_masks': grounding_masks,
                                'dataset': dataset.get_name(),
                                'test_class': meta_obj_test.get('object_class_name'),
                                'exp_str': meta_obj_test.get('exp_str') if 'exp_str' in meta_obj_test else None,
                                })
        
                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

