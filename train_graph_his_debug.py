from __future__ import  absolute_import
import os

import numpy as np
import matplotlib
from tqdm import tqdm
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from cardiac_uda import Seg_Cardiac_UDA_Dataset

from data.dataset import inverse_normalize
from data.fetus_dataset import fetus_Dataset, collate_fn

from utils.config import opt
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from utils.boxlist import BoxList
from utils.gpu_tools import get_world_size, get_global_rank, get_local_rank, get_master_ip

from utils.distributed import get_rank, synchronize, reduce_loss_dict, DistributedSampler, all_gather
from utils.graph_config import _C as graph_opt
from utils.build_opt import make_optimizer, make_lr_scheduler

import torch.nn.functional as F

from model.topograph_net import FCOSPostprocessor, Topograph
# from model.topograph_net_vgg16 import Topograph, FCOSPostprocessor
from model.graph_matching import build_graph_matching_head
from model.substructure_matching import substructure_matching_sinkhorn, substructure_matching_L2
from utils.slice import slice_tensor
from torch.utils.tensorboard import SummaryWriter 
from skimage import exposure
import time
from torchvision.transforms import ToPILImage
from data.fetus_dataset import annnotations_convert

import resource
import wandb

import warnings

from utils.histogram_util import match_histograms
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
print(starttime[:19])
writer = SummaryWriter(log_dir="log_m_h/"+starttime[:19]+opt.description,comment=starttime[:19],flush_secs=30)

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        root = 'Cardiac_UDA_' 
        infos = np.load(f'{root}/info.npy', allow_pickle=True).item()
        self.label_num = len(annnotations_convert[opt.slices[0]])
        self.opt.n_class = self.label_num + 1
        opt.n_class = self.label_num + 1
        graph_opt.MODEL.FCOS.NUM_CLASSES = self.label_num + 1
        opt.n_class = self.label_num + 1

        print('Load Fetus Dataset')
        if self.opt.selected_source_hospital[0] in ['Hospital_1','Hospital_2','Hospital_3']:
            train_source_set = fetus_Dataset(self.opt, operation='train')
        else:
            self.opt.n_class = 5
            opt.n_class = 5
            self.label_num = 4
            graph_opt.MODEL.FCOS.NUM_CLASSES = self.label_num + 1
            opt.n_class = self.label_num + 1
            train_source_set = Seg_Cardiac_UDA_Dataset(infos, root, is_train=True, set_select=self.opt.selected_source_hospital, view_num=['4'], seg_parts=True)
        self.train_source_dataloader = DataLoader(train_source_set,
                                        collate_fn = collate_fn(opt),
                                        batch_size=2,
                                        shuffle=True,
                                        num_workers=self.opt.num_workers,
                                        drop_last=True)

        if self.opt.selected_target_hospital[0] in ['Hospital_1','Hospital_2','Hospital_3']:
            train_target_set = fetus_Dataset(self.opt, operation='train', domain='Target')
            vaildset = fetus_Dataset(self.opt, operation='valid', domain='Target')
            testset  = fetus_Dataset(self.opt, operation='test', domain='Target')
        else:
            train_target_set = Seg_Cardiac_UDA_Dataset(infos, root, is_train=True, set_select=self.opt.selected_target_hospital, view_num=['4'], seg_parts=True)
            vaildset = Seg_Cardiac_UDA_Dataset(infos, root, is_train=False, set_select=self.opt.selected_target_hospital, view_num=['4'], seg_parts=True)
            testset  = Seg_Cardiac_UDA_Dataset(infos, root, is_train=False, set_select=self.opt.selected_target_hospital, view_num=['4'], seg_parts=True)
        
        self.train_target_dataloader = DataLoader(train_target_set,
                                        collate_fn = collate_fn(opt),
                                        batch_size=2,
                                        shuffle=True,
                                        num_workers=self.opt.num_workers,
                                        drop_last=True)
        
        self.vaild_dataloader = DataLoader(vaildset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,)
        
        self.test_dataloader = DataLoader(testset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=self.opt.test_num_workers,
                                        shuffle=False,)

        print('Build BackBone Network & Graph Matching Module')
        self.model = Topograph(self.opt, Topograph_m=True).to(device=opt.device)
        self.graph_matching = build_graph_matching_head(graph_opt, self.opt.out_channel).to(device=opt.device)
        self.postprocessor = FCOSPostprocessor(opt)
        print('Model Construct Completed')

        self.fpn_strides = opt.fpn_strides

        print('Build Optimizer & Scheduler for BackBone and Graph Matching')
        self.optimizer = {}
        self.scheduler = {}
        
        self.optimizer["backbone"] = make_optimizer(graph_opt, self.model, name='backbone')
        self.optimizer["middle_head"] = make_optimizer(graph_opt, self.graph_matching, name='backbone')
        self.scheduler["backbone"] = make_lr_scheduler(graph_opt, self.optimizer["backbone"], name='middle_head')
        self.scheduler["middle_head"] = make_lr_scheduler(graph_opt, self.optimizer["middle_head"], name='middle_head')

        self.toPIL = ToPILImage()

        if self.opt.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.opt.local_rank],
                output_device=self.opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,)

            self.graph_matching = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.graph_matching)
            self.graph_matching = torch.nn.parallel.DistributedDataParallel(
                self.graph_matching,
                device_ids=[self.opt.local_rank],
                output_device=self.opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,)

    def train(self):
        if self.opt.load_path:
            self.load(self.opt.load_path)
            print('load pretrained model from %s' % self.opt.load_path)

        best_map = 0
        lr_ = self.opt.lr
        for epoch in range(self.opt.epoch):
            self.model.train()

            len_source = len(self.train_source_dataloader)
            len_target = len(self.train_target_dataloader)
            max_len = max(len_source, len_target)

            source_iter = iter(self.train_source_dataloader)
            target_iter = iter(self.train_target_dataloader)

            for _ in tqdm(range(max_len)):
                # if step == max_step:
                #     train_target_dataloader_use = train_target_dataloader_second

                try:
                    imgs_src, targets_src, _ = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.train_source_dataloader)
                    imgs_src, targets_src, _ = next(source_iter)

                try:
                    imgs_tgt, _, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.train_target_dataloader)
                    imgs_tgt, _, _ = next(target_iter)
                
                targets_src = [target.to(device=opt.device) for target in targets_src]

                # sp Tranfer (histogram match)
                # two ways, first implement with exposure and second one implement with match_histograms in /v3/utils/histogram_util.py
                # imgs_src = torch.stack([
                #     torch.from_numpy(match_histograms(img.permute(1, 2, 0).numpy(), img_target.permute(1, 2, 0).numpy())).permute(2, 0, 1)
                #     for img, img_target in zip(imgs_src.tensors, imgs_tgt.tensors)
                # ], dim=0).float()
                imgs_tgt.tensors = torch.stack([
                    torch.from_numpy(exposure.match_histograms(img.permute(1, 2, 0).numpy(), img_target.permute(1, 2, 0).numpy())).permute(2, 0, 1)
                    for img, img_target in zip(imgs_tgt.tensors, imgs_src.tensors)
                ], dim=0).float()
                
                (features_src, _, _, _), _, losses = \
                    self.model(imgs_src.tensors.to(device=opt.device), image_sizes=None, targets=targets_src, train=True, domain='Source')
                (features_tgt, cls_pred_tgt, box_pred_tgt, center_pred_tgt), _ = \
                    self.model(imgs_tgt.tensors.to(device=opt.device), image_sizes=None, targets=None, train=True, domain='Target')
                
                score_maps_tgt = self.model._forward_target(cls_pred_tgt, box_pred_tgt, center_pred_tgt)

                (_, _), middle_head_loss = self.graph_matching(None, (features_src, features_tgt), targets=targets_src, score_maps=score_maps_tgt)
                
                loss_sub_m = torch.tensor(0,device=opt.device,dtype=float)
                if epoch >= opt.match_start_epoch:
                    features_t_slice = slice_tensor(features_tgt)
                    cls_pred_t_slice = slice_tensor(cls_pred_tgt)
                    box_pred_t_slice = slice_tensor(box_pred_tgt)
                    center_pred_t_slice = slice_tensor(center_pred_tgt)
                    
                    for i in range(len(targets_src)):
                        location = self.compute_location(features_t_slice[i])
                        boxes = self.postprocessor(
                            location, cls_pred_t_slice[i], box_pred_t_slice[i], center_pred_t_slice[i], imgs_tgt.sizes[i]
                        )
                        label = boxes[0].fields['labels']
                        # Specifies whether all class nodes are owned
                        unique_v = set(label.tolist())
                        if len(unique_v) == self.label_num and set(range(1, self.label_num + 1)).issubset(unique_v):
                            loss_sub_m += substructure_matching_L2(targets_src[i], boxes[0],self.label_num)
                        else:
                            continue
                
                loss_cls = losses['loss_cls'].mean()
                loss_box = losses['loss_box'].mean()
                loss_center = losses['loss_center'].mean()
                backbone_loss = loss_cls + loss_box + loss_center

                loss_matching = sum(loss for loss in middle_head_loss.values())
                ## loss_matching including loss_classification and loss_matching
                # loss_matching = torch.tensor(0,device=opt.device,dtype=float)
                overall_loss = backbone_loss + loss_matching * 0.5 + loss_sub_m * 0.5
                 
                for opt_k in self.optimizer:
                    self.optimizer[opt_k].zero_grad()
                
                overall_loss.backward()

                for opt_k in self.optimizer:
                    self.optimizer[opt_k].step()

            if not isinstance(loss_sub_m, torch.Tensor):
                loss_sub_m = torch.tensor(loss_sub_m,device=opt.device,dtype=float)
                
            eval_result = self.eval(self.vaild_dataloader, test_num=self.opt.test_num)
            log_info = 'epoch:{}, map:{},loss:{},backbone_loss:{},loss_matching:{},loss_sub_m:{},ap:{}'.format(str(epoch),
                                                         str(eval_result['map']),
                                                         str(round(overall_loss.item(),4)),
                                                         str(round(backbone_loss.item(),4)),
                                                         str(round(loss_matching.item(),4)),
                                                         str(round(loss_sub_m.item(),4)),
                                                         str(eval_result['ap']),
                                                        )
            
            writer.add_scalar('mAP', eval_result['map'], global_step=epoch, walltime=None)
            writer.add_scalar('overall_loss', overall_loss, global_step=epoch, walltime=None)
            writer.add_scalar('backbone_loss', backbone_loss, global_step=epoch, walltime=None)
            writer.add_scalar('loss_matching', loss_matching, global_step=epoch, walltime=None)
            writer.add_scalar('loss_sub_m', loss_sub_m, global_step=epoch, walltime=None)
            print(log_info)

            # Update optimizers with scheduler
            for scheduler_k in self.scheduler:
                self.scheduler[scheduler_k].step()
            
            if eval_result['map'] > best_map and eval_result['map'] > 0.4:
                best_map = eval_result['map']
                best_path = self.save(best_map=best_map)

            if epoch > opt.epoch: 
                self.load(best_path)
                test_result = self.eval(self.test_dataloader, test_num=self.opt.test_num)
                log_info = 'final test ---> epoch:{}, map:{},loss:{}'.format(str(epoch),
                                                                             str(test_result['map']),
                                                                             str(overall_loss.item()))
                print(log_info)
                break

    def accumulate_predictions(self, predictions):
        all_predictions = all_gather(predictions)

        if get_rank() != 0:
            return

        predictions = {}

        for p in all_predictions:
            predictions.update(p)

        ids = list(sorted(predictions.keys()))

        if len(ids) != ids[-1] + 1:
            print('Evaluation results is not contiguous')

        predictions = [predictions[i] for i in ids]

        return predictions

    @torch.no_grad()
    def eval(self, dataloader, test_num=10000):
        self.model.eval()
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        for ids, (imgs, gt_targets, ids) in tqdm(enumerate(dataloader)):

            preds = {}
            imgs = imgs.tensors.to(device=opt.device)
            imgs = F.interpolate(imgs, size=(896, 896), mode='bilinear', align_corners=False)

            gt_targets = [target.to('cpu') for target in gt_targets]

            pred, _ = self.model(imgs, imgs.shape[-2:], train=False)
            pred = [p.to('cpu') for p in pred]
            preds = pred

            for idx, pred in enumerate(preds):
                _pred_bboxes = pred.box.numpy()
                _pred_labels = pred.fields['labels'].numpy()
                _pred_scores = pred.fields['scores'].numpy()
                _gt_bboxes_ = gt_targets[idx].box.numpy()
                _gt_labels_ = gt_targets[idx].fields['labels'].numpy()

            if _pred_bboxes.shape[0] == 0:
                continue
            else:
                pred_bboxes += [_pred_bboxes]
                pred_labels += [_pred_labels]
                pred_scores += [_pred_scores]

                gt_bboxes += [_gt_bboxes_]
                gt_labels += [_gt_labels_]
                # gt_difficults.append(gt_difficults_)

            if ids == test_num: break

        gt_difficults = None
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
        return result

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.model.state_dict()
        save_dict['config'] = opt._state_dict()

        if save_optimizer:
            for opt_k in self.optimizer:
                save_dict['optimizer'][opt_k] = self.optimizer[opt_k].state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/'+opt.model_name+'%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False,):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.model.load_state_dict(state_dict)
            return self
        if parse_opt:
            self.opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations
    
    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location


def main(rank, opt):

    try:
        opt.local_rank
    except AttributeError:
        opt.global_rank = rank
        opt.local_rank = opt.enable_GPUs_id[rank]
    else:
        if opt.distributed:
            opt.global_rank = rank
            opt.local_rank = opt.enable_GPUs_id[rank]

    if opt.distributed:
        torch.cuda.set_device(int(opt.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt.init_method,
                                             world_size=opt.world_size,
                                             rank=opt.global_rank,
                                             group_name='mtorch'
                                             )

        print('using GPU {}-{} for training'.format(
            int(opt.global_rank), int(opt.local_rank)
            ))

        if opt.local_rank == opt.enable_GPUs_id[0]:
            wandb_init()

    if torch.cuda.is_available(): 
        opt.device = torch.device("cuda:{}".format(opt.local_rank))
    else: 
        opt.device = 'cpu'
    
    Train_ = Trainer(opt)
    Train_.train()
    

if __name__ == '__main__':

    # setting distributed configurations
    opt.world_size = len(opt.enable_GPUs_id)
    opt.init_method = f"tcp://{get_master_ip()}:{23455}"
    opt.distributed = True if opt.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and opt.distributed:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=opt.world_size, args=(opt,))
    else:
        # multiple processes have been launched by openmpi
        opt.local_rank = opt.enable_GPUs_id[0]
        opt.global_rank = opt.enable_GPUs_id[0]

        main(opt.local_rank, opt)
