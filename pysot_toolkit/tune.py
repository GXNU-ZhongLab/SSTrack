from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from toolkit.datasets import DatasetFactory, OTBDataset, UAVDataset, LaSOTDataset, \
    VOTDataset, NFSDataset, VOTLTDataset
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
    EAOBenchmark, F1Benchmark

import optuna
import logging
from bbox import get_axis_aligned_bbox
from lib.config.siampin.config import cfg
from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target
from lib.models.stark.siampin import build_TransT
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
import importlib
import torch.nn.functional as F


class TransT(BaseTracker):
    def __init__(self, params):
        super(TransT, self).__init__(params)
        network = build_TransT(params)
        network.load_state_dict(torch.load(args.snapshot, map_location='cpu')['net'], strict=True)

        self.cfg = params
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = False
        self.z_dict1 = {}

        self.window_penalty = params.TEST.WINDOW_PENALTY

    def initialize(self, image, info: dict):
        # 余弦窗
        hanning = np.hanning(self.network.feat_sz_s)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()

        z_patch_arr, _, z_mask_arr = sample_target(image, info['init_bbox'], self.params.TEST.TEMPLATE_FACTOR,
                                                   output_sz=self.params.TEST.TEMPLATE_SIZE)
        template = self.preprocessor.process(z_patch_arr, z_mask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.init(template)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, dataset=None, video=None, frame=0):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_mask_arr = sample_target(image, self.state, self.params.TEST.SEARCH_FACTOR,
                                                   output_sz=self.params.TEST.SEARCH_SIZE)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_mask_arr)

        with torch.no_grad():
            out_dict = self.network.track(search)

        score = self._convert_score(out_dict['pred_logits'])  # numpy 类型  (1024,)
        pred_boxes = self._convert_bbox(out_dict['pred_boxes'])  # tensor 类型  (4, 1024)

        # window penalty
        pscore = score * (1 - self.window_penalty) + self.window * self.window_penalty
        best_idx = np.argmax(pscore)
        pred_boxes = pred_boxes[:, best_idx][np.newaxis, :]

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.TEST.SEARCH_SIZE / resize_factor).tolist()  # 映射到原图坐标(cx, cy, w, h)

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)  # (x1, y1, w, h)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,
                    "best_score": score[best_idx],
                    }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.TEST.SEARCH_SIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):
        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu()
        # delta = delta.data.cpu().numpy()
        return delta


def eval(dataset, tracker_name):
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    # root = os.path.join(root, dataset)
    tracker_dir = "./"
    trackers = [tracker_name]
    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'NFS' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao
    elif 'VOT2018-LT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=1) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=False)

    return 0

# fitness function
def objective(trial):
    # different params
    cfg.TEST.WINDOW_PENALTY = trial.suggest_uniform('WINDOW_PENALTY', 0.1, 0.8)
    
    # rebuild tracker
    tracker = TransT(cfg)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    tracker_name = os.path.join('tune_results',args.dataset, model_name + \
                    '_win-{:.3f}'.format(cfg.TEST.WINDOW_PENALTY))
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    # gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                    gt_bbox = np.array(gt_bbox)
                    # cx = np.mean(gt_bbox_[0::2])
                    # cy = np.mean(gt_bbox_[1::2])
                    x1 = min(gt_bbox[0::2])
                    x2 = max(gt_bbox[0::2])
                    y1 = min(gt_bbox[1::2])
                    y2 = max(gt_bbox[1::2])
                    A1 = np.linalg.norm(gt_bbox[0:2] - gt_bbox[2:4]) * \
                         np.linalg.norm(gt_bbox[2:4] - gt_bbox[4:6])
                    A2 = (x2 - x1) * (y2 - y1)
                    s = np.sqrt(A1 / A2)
                    w = s * (x2 - x1) + 1
                    h = s * (y2 - y1) + 1

                    gt_bbox = [x1, y1, w, h]

                    gt_bbox_ = {'init_bbox': gt_bbox}  # {'init_bbox': [x1, y1, w, h]}
                    tracker.initialize(img, gt_bbox_)

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    # pred_bbox = outputs['bbox']
                    pred_bbox = outputs['target_bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(tracker_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))

        eao = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_penalty: {:1.17f}, EAO: {:1.3f}".format(model_name, cfg.TEST.WINDOW_PENALTY, eao)
        logging.getLogger().info(info)
        print(info)
        return eao

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    # gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                    gt_bbox_ = {'init_bbox': gt_bbox}  # {'init_bbox': [x1, y1, w, h]}
                    tracker.initialize(img, gt_bbox_)

                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox['init_bbox'])
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path, '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))

        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_penalty: {:1.17f}, AUC: {:1.3f}".format(model_name, cfg.TEST.WINDOW_PENALTY, auc)
        logging.getLogger().info(info)
        print(info)
        return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tuning for SiamBAN')
    parser.add_argument('--script_name', default='siampin', type=str, help='script_name')
    parser.add_argument('--dataset', default='VOT2019', type=str, help='dataset')
    parser.add_argument('--cfg_file',
                        default='/media/disk1/projects/service/32.convTrans/experiments/siampin/baseline.yaml',
                        type=str, help='config file')
    parser.add_argument('--snapshot',
                        default='/media/disk1/projects/service/32.convTrans/checkpoints/train/siampin/baseline/TransT_ep0133.pth.tar',
                        type=str,
                        help='snapshot of models to eval')
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")

    args = parser.parse_args()

    torch.set_num_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # update the default configs with config file
    if not os.path.exists(args.cfg_file):
        raise ValueError("%s doesn't exist." % args.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % args.script_name)  # 读取lib/config/下的配置文件
    cfg = config_module.cfg  # cfg: config.py配置文件
    config_module.update_config_from_file(args.cfg_file)  # 更新配置文件，读取experiments/下的配置文件

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # Eval dataset
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), '../testing_dataset'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset_eval = OTBDataset(args.dataset, root)
    elif 'LaSOT' == args.dataset:
        dataset_eval = LaSOTDataset(args.dataset, root)
    elif 'UAV' in args.dataset:
        dataset_eval = UAVDataset(args.dataset, root)
    elif 'NFS' in args.dataset:
        dataset_eval = NFSDataset(args.dataset, root)
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset_eval = VOTDataset(args.dataset, root)
    elif 'VOT2018-LT' == args.dataset:
        dataset_eval = VOTLTDataset(args.dataset, root)
    
    tune_result = os.path.join('tune_results', args.dataset)
    if not os.path.isdir(tune_result):
        os.makedirs(tune_result)
    log_path = os.path.join(tune_result, (args.snapshot).split('/')[-1].split('.')[0] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
