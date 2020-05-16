import torch
from torch import nn
from loss.losses import *
from loss.utils import *
import numpy as np


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, pred, gt):
        pred_hm, pred_wh, pred_offset = pred
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        gt_nonpad_mask = gt_classes.gt(0)

        # print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pred_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1]
            ]).view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)

            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6)

        ### IOU LOSS ###
        # output_h, output_w = pred_hm.shape[-2:]
        # b, _, h, w = imgs.shape
        # location = map2coords(output_h, output_w, self.down_stride).cuda()
        #
        # location = location.view(output_h, output_w, 2)
        # pred_offset = pred_offset.permute(0, 2, 3, 1)
        # pred_wh = pred_wh.permute(0, 2, 3, 1)
        # iou_loss = cls_loss.new_tensor(0.)
        # for batch in range(b):
        #     ct = infos[batch]['ct']
        #     xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = [[] for _ in range(6)]
        #     for i, cls in enumerate(gt_classes[batch][gt_nonpad_mask[batch]]):
        #         ct_int = ct[i]
        #         xs.append(location[ct_int[1], ct_int[0], 0])
        #         ys.append(location[ct_int[1], ct_int[0], 1])
        #         pos_w.append(pred_wh[batch, ct_int[1], ct_int[0], 0])
        #         pos_h.append(pred_wh[batch, ct_int[1], ct_int[0], 1])
        #         pos_offset_x.append(pred_offset[batch, ct_int[1], ct_int[0], 0])
        #         pos_offset_y.append(pred_offset[batch, ct_int[1], ct_int[0], 1])
        #     xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = \
        #         [torch.stack(i) for i in [xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y]]
        #     #####
        #
        #     det_boxes = torch.stack([
        #         xs - pos_w / 2 + pos_offset_x,
        #         ys - pos_h / 2 + pos_offset_y,
        #         xs + pos_w / 2 + pos_offset_x,
        #         ys + pos_h / 2 + pos_offset_y
        #     ]).T.round()
        #
        #     iou_loss += self.iou_loss(det_boxes, gt_boxes[batch][gt_nonpad_mask[batch]])

        # return cls_loss * self.alpha,  iou_loss / b * self.beta



