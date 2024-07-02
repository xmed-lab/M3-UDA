from __future__ import  absolute_import
import os
import random

import numpy as np
import matplotlib
from sklearn.manifold import TSNE
from tqdm import tqdm
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2

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

from model.topograph_net import Topograph
from model.graph_matching import build_graph_matching_head

from utils.histogram_util import match_histogram

import matplotlib.pyplot as plt
from sklearn import datasets

import resource
import wandb

import warnings
from data.fetus_dataset import annnotations_convert
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


seed_num = 6666
np.set_printoptions(threshold=np.inf)
np.random.seed(seed_num)
random.seed(seed_num)

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        print('Load Fetus Dataset')
        self.opt = opt
        self.label_num = len(annnotations_convert[opt.slices[0]])
        self.opt.n_class = self.label_num + 1
        opt.n_class = self.label_num + 1
        graph_opt.MODEL.FCOS.NUM_CLASSES = self.label_num + 1
        opt.n_class = self.label_num + 1

        testset  = fetus_Dataset(self.opt, operation='test', domain='Target')
        self.test_dataloader = DataLoader(testset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False,)

        print('Build BackBone Network & Graph Matching Module')
        self.model = Topograph(self.opt, Topograph_m=True).to(device=opt.device)
        device = next(self.model.parameters()).device
        print(f"Model is on device: {device}")
        # self.graph_matching = build_graph_matching_head(graph_opt, self.opt.out_channel).to(device=opt.device)
        print('Model Construct Completed')
            
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


    def test(self):
        self.load(opt.test_checkpoint_name)
        test_result = self.eval(self.test_dataloader, test_num=self.opt.test_num)
        log_info = 'map:{}'.format(str(test_result['map']))
        print(str(test_result['ap']))
        print(log_info)


    @torch.no_grad()
    def eval(self, dataloader, test_num=10000):
        self.model.eval()
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        idx = 0
        features_tsne = list()
        labels_tsne = list()
        for ids, (imgs, gt_targets, ids) in tqdm(enumerate(dataloader)):

            preds = {}
            imgs = imgs.tensors.to(device=opt.device)

            gt_targets = [target.to('cpu') for target in gt_targets]
            with torch.no_grad():
                pred, cls_pred = self.model(imgs, imgs.shape[-2:], train=False)

            label = torch.arange(1, 10)
            cls_pred = cls_pred[4].squeeze(dim=0)
            cls_pred = cls_pred.cpu()
            cls_pred = cls_pred.view(cls_pred.shape[0], -1)

            features_tsne.append(cls_pred)
            labels_tsne.append(label)

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
                bbox = _pred_bboxes
                label = _pred_labels
                img = imgs.squeeze(0)
                image_p = T.ToPILImage()(img)
                bbox = bbox.astype(int)
                font = ImageFont.truetype(font='Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(image_p)[1] + 15).astype('int32'))
                arr = []
                for i in range(_pred_labels.min(),_pred_labels.max()+1):
                    l = np.where(_pred_labels == i)
                    if l is not None and l[0].size != 0:
                        arr.append(l[0][0])
                        if i == 9 and l[0].shape[0] > 1:
                            arr.append(l[0][1])

                
                for i in arr:
                    draw = ImageDraw.Draw(image_p)
                    # 获取label长宽
                    label_size1 = draw.textsize(str(label[i]), font)
                    # 设置label起点
                    text_origin1 = np.array([bbox[i][0], bbox[i][1] - label_size1[1]])
                    # 绘制矩形框，加入label文本
                    draw.rectangle([bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]],outline='red',width=2)
                    draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
                    draw.text(text_origin1, str(label[i]), fill=(255, 255, 255), font=font)
                gt_box = gt_targets[0].box
                gt_label = gt_targets[0].fields['labels']
                for i in range(gt_box.shape[0]):
                    draw = ImageDraw.Draw(image_p)
                    # 获取label长宽
                    label_size1 = draw.textsize(str(gt_label[i].item()), font)
                    # 设置label起点
                    text_origin1 = np.array([gt_box[i][0], gt_box[i][1] - label_size1[1]])
                    # 绘制矩形框，加入label文本
                    draw.rectangle([gt_box[i][0], gt_box[i][1], gt_box[i][2], gt_box[i][3]],outline='blue',width=1)
                    draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='blue')
                    draw.text(text_origin1, str(gt_label[i].item()), fill=(255, 255, 255), font=font)
                image_p.save('imgs/'+ str(ids[0]) +'.jpg')
                pred_bboxes += [_pred_bboxes]
                pred_labels += [_pred_labels]
                pred_scores += [_pred_scores]

                gt_bboxes += [_gt_bboxes_]
                gt_labels += [_gt_labels_]
                # gt_difficults.append(gt_difficults_)

            # if ids == test_num: break
            # print(pred.box.numpy())
            # break

        gt_difficults = None
        
        
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
        # 获取所有张量的第二个维度大小
        size_dim1 = min(tensor.size(1) for tensor in features_tsne)
        # 调整所有张量的第二个维度大小为相同的值
        features_tsne_adjusted = [tensor[:, :size_dim1] for tensor in features_tsne]
        # 将张量列表转换为 NumPy 数组并进行拼接
        features_tsne = np.concatenate([tensor.numpy() for tensor in features_tsne_adjusted], axis=0)
        labels_tsne = np.concatenate([tensor.numpy() for tensor in labels_tsne], axis=0)
        # features_tsne = torch.cat(features_tsne, dim=0)
        # labels_tsne = torch.cat(labels_tsne, dim=0)
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
        features_tsne = tsne.fit_transform(features_tsne)
        # 根据标签使用不同的颜色进行绘制
        current_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # 打印颜色循环
        print("当前颜色循环:", current_color_cycle)
        for i in np.unique(labels_tsne):
            plt.scatter(features_tsne[labels_tsne == i, 0], features_tsne[labels_tsne == i, 1], label=f'Class {i}')
        # plt.title('T-SNE Visualization of baseline Features')
        plt.xlabel('T-SNE Dimension 1')
        plt.ylabel('T-SNE Dimension 2')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig('our_3.png')
        return result
    

def main(rank, opt):
    if torch.cuda.is_available(): 
        opt.device = torch.device("cuda:{}".format(opt.local_rank))
    else: 
        opt.device = 'cpu'
    
    Train_ = Trainer(opt)
    Train_.test()

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