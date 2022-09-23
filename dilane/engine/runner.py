from datetime import date
import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from dilane.models.registry import build_net
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from dilane.datasets import build_dataloader
from dilane.utils.recorder import build_recorder
from dilane.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        # self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            # self.grad_cam(epoch, train_loader) #可视化
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                # end = time.time()
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                # batch_time = time.time()-end
                # self.recorder.batch_time.update(batch_time)
                # if i % 100 == 0:
                #     self.recorder.record('validate')
                predictions.extend(output)
            if self.cfg.view:
                labels = self.net.module.heads.get_labels(data['lane_line'])
                self.val_loader.dataset.view(output, data['meta'], labels)

        metric = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)


#------------------Grad-cam----------------------------------
    def cam_show_img(self, img, feature_map, grads, out_dir):
        H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
        for i in range(feature_map.shape[0]):
            fmap = feature_map[i, :, :]
            fmap = fmap / np.absolute(fmap).max()
            # fmap = cv2.resize(fmap, (W, H))
            fmap = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_RAINBOW)
            cv2.imshow('fmap_{}'.format(i),  fmap)

        grads = grads.reshape([grads.shape[0],-1])					# 5
        weights = np.mean(grads, axis=1)							# 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							# 7
        # cam = np.maximum(cam, 0)
        cam = cam / np.absolute(cam).max()
        cam = cv2.resize(cam, (W, H))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        path_cam_img = os.path.join(out_dir, "cam.jpg")
        cv2.imshow('cam_img', cam_img)
        cv2.imwrite(path_cam_img, heatmap)
        cv2.waitKey(0)

    def grad_cam(self, epoch, train_loader):
        fmap_block = list()
        grad_block = list()
        output_dir = './cam'
        def farward_hook(module, input, output):
            return fmap_block.append(output[0])
        def backward_hook(module, grad_in, grad_out):
            return grad_block.append(grad_out[0].detach())

        self.net.module.neck.fpn_convs[0].register_forward_hook(farward_hook)
        self.net.module.neck.fpn_convs[0].register_forward_hook(backward_hook)
        self.net.train()
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            self.recorder.step += 1
            img = data['img'][0].permute(1,2,0)
            data = self.to_cuda(data)
            self.net.zero_grad()
            output = self.net(data)
            loss = output['loss'].sum()
            loss.backward()

            img = img.cpu().data.numpy()
            grads_val = grad_block[-1].cpu().data.numpy().squeeze()
            fmap = fmap_block[-1].cpu().data.numpy().squeeze()

            # 保存cam图片
            self.cam_show_img(img, fmap, grads_val, output_dir)
            if(len(fmap_block)>10):
                fmap_block.pop(0)
                grad_block.pop(0)
