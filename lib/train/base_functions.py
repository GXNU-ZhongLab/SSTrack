import os
import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
from lib.train.dataset import Davis, YouTubeVOS, Got10kVOS, LasotVOS
from lib.train.dataset import Refer_YouTubeVOS, TNL2k, TNL2k_Lang, Lasot_Lang, OTB_Lang, RefCOCOSeq
from lib.train.dataset import VisEvent, LasHeR, DepthTrack, ARKit


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR,
                                   'grounding': cfg.DATA.GROUNDING.FACTOR,
                                   }
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE,
                          'grounding': cfg.DATA.GROUNDING.SIZE,
                          }
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'grounding': cfg.DATA.GROUNDING.CENTER_JITTER,
                                     }
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'grounding': cfg.DATA.GROUNDING.SCALE_JITTER,
                                    }
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET",
                        "DAVIS", "YouTubeVOS2018", "YouTubeVOS2018_val", "YouTubeVOS2019", "YouTubeVOS2019_val", "Got10kVOS", "LasotVOS",
                        "LASOT_Lang", "TNL2K", "TNL2K_Lang", "OTB_Lang", "Refer_YouTubeVOS", "RefCOCO14",
                        "DepthTrack_train", "DepthTrack_val", "art_train", "art_test", "LasHeR_all", "LasHeR_train", "LasHeR_val", "VisEvent",
                        ]
        # Tracking Task
        if name == "TNL2K":
            datasets.append(TNL2k(settings.env.tnl2k_dir, split='train'))
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        
        # Segmentation Task
        if name == "DAVIS":
            datasets.append(Davis(settings.env.davis_dir, version="2017", image_loader=image_loader,
                                        multiobj=False, split='train'))
        if name == "YouTubeVOS2018":
            datasets.append(YouTubeVOS(settings.env.youtubevos_dir, version="2018", image_loader=image_loader,
                                        multiobj=False, split='jjtrain'))
        if name == "YouTubeVOS2018_val":
            datasets.append(YouTubeVOS(settings.env.youtubevos_dir, version="2018", image_loader=image_loader,
                                        multiobj=False, split='jjvalid'))
        if name == "YouTubeVOS2019":
            datasets.append(YouTubeVOS(settings.env.youtubevos_dir, version="2019", image_loader=image_loader,
                                        multiobj=False, split='jjtrain'))
        if name == "YouTubeVOS2019_val":
            datasets.append(YouTubeVOS(settings.env.youtubevos_dir, version="2019", image_loader=image_loader,
                                        multiobj=False, split='jjvalid'))
        if name == "Got10kVOS":
            anno_path = os.path.join(settings.env.tracking_masks_dir, "got10k_masks")
            datasets.append(Got10kVOS(anno_path=anno_path, split='vottrain'))
        if name == "LasotVOS":
            anno_path = os.path.join(settings.env.tracking_masks_dir, "lasot_masks")
            datasets.append(LasotVOS(anno_path=anno_path, split='train'))
        
        # Visual-Language Task
        if name == "TNL2K_Lang":
            datasets.append(TNL2k_Lang(settings.env.tnl2k_dir, split='train'))
        if name == "LASOT_Lang":
            datasets.append(Lasot_Lang(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "OTB_Lang":
            datasets.append(OTB_Lang(settings.env.otb_lang_dir, split='train', image_loader=image_loader))
        if name == "Refer_YouTubeVOS":
            datasets.append(Refer_YouTubeVOS(settings.env.refer_youtubevos_dir, version="2019", 
                                                multiobj=False, split='jjtrain'))
        if name == "RefCOCO14":
            datasets.append(RefCOCOSeq(settings.env.ref_coco_dir, refcoco_type="refcoco-unc", version="2014", image_loader=image_loader))

        # Multi-modal tracking tasks
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='RGBD', split='train'))  # rgbcolormap
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='RGBD', split='val'))  # rgbcolormap
        if name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='RGBT', split='all'))  # rgbrgb
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='RGBT', split='train'))  # rgbrgb
        if name == "LasHeR_val":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='RGBT', split='val'))  # rgbrgb
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='RGBE', split='train'))  # rgbrgb
        if name == "art_train":
            datasets.append(ARKit(settings.env.arkit_dir, split='train', image_loader=image_loader, dtype='RGBD'))
        if name == "art_test":
            datasets.append(ARKit(settings.env.arkit_dir, split='test', image_loader=image_loader, dtype='RGBD'))
        
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    )

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
                                    )

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       search_transform=transform_val,
                                                       joint_transform=transform_joint,
                                                       settings=settings,)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     search_transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings,)

    # Train sampler and loader
    settings.num_grounding = getattr(cfg.DATA.GROUNDING, "NUMBER", 1)
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    settings.num_grounding = getattr(cfg.DATA.GROUNDING, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    template_sample_range = getattr(cfg.DATA.TEMPLATE, "SAMPLE_INTERVAL", 0)
    
    print("sampler_mode: ", sampler_mode)
    dataset_train = sampler.SelfSupervised_Sampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, 
                                            num_grounding_frames=settings.num_grounding, 
                                            processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls,
                                            template_sample_range=template_sample_range)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler,
                            )

    # Validation samplers and loaders
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.SelfSupervised_Sampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template,
                                            num_grounding_frames=settings.num_grounding,
                                            processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls,
                                            template_sample_range=template_sample_range)
    
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg, settings):
    tracker_name = settings.script_name

    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
    ]

    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print("Learnable parameters: ", n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
