import torch
from torch import nn
import numpy as np

B = 2
NB_CLASSES = 10

def xywhc2label(bboxs):
    label = np.zeros((7, 7, 5*B+NB_CLASSES))
    for x, y, w, h, c in bboxs:
        x_grid = int(x//(1/7))
        y_grid = int(y//(1/7))
        label[x_grid][y_grid][0:5] = np.array([x, y, w, h, 1])
        label[x_grid][y_grid][5:10] = np.array([x, y, w, h, 1])
        label[x_grid][y_grid][10:10+c-1] = 1
    return label


def pred2xywhcc(pred):
    # pred is 1*7*7*(5*B+C) tensor
    # convert to numpy array
    # pred = pred.detach().numpy()
    bbox = torch.zeros((7*7*2, 5+10))  # 98 * 15
    for x in range(7):
        for y in range(7):
            # bbox1
            bbox[2*(x*7+y), 0:4] = pred[x, y, 0:4]
            bbox[2*(x*7+y), 4] = pred[x, y, 4]
            bbox[2*(x*7+y), 5:] = pred[x, y, 10:]
            # bbox2
            bbox[2*(x*7+y)+1, 0:4] = pred[x, y, 5:9]
            bbox[2*(x*7+y)+1, 4] = pred[x, y, 9]
            bbox[2*(x*7+y)+1, 5:] = pred[x, y, 10:]
    # NMS
    xywhcc = nms(bbox)
    return xywhcc


def nms(bbox, conf_thresh=0.1, iou_thresh=0.3):
    # Non-Maximum Suppression
    bbox_prob = bbox[:, 5:].clone()
    bbox_conf = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)
    bbox_cls_spec_conf = bbox_conf*bbox_prob
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    for c in range(10):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i], c] != 0:
                for j in range(i+1, 98):
                    if bbox_cls_spec_conf[rank[j], c] != 0:
                        iou = calculate_iou(
                            bbox[rank[i], 0:4], bbox[rank[j], 0:4])
                        if iou > iou_thresh:
                            bbox_cls_spec_conf[rank[j], c] = 0

    # exclude cls-specific confidence score=0
    bbox = bbox[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(
        bbox_cls_spec_conf, dim=1).values > 0]

    res = torch.ones((bbox.size()[0], 6))
    res[:, 0:4] = bbox[:, 0:4]  # bbox coord
    res[:, 6] = torch.argmax(bbox[:, 5:], dim=1).int()  # bbox class
    # bbox class-specific confidence scores
    res[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values
    return res


def calculate_iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = bbox1.cpu().detach().numpy(
    ).tolist(), bbox2.cpu().detach().numpy().tolist()
    # print(bbox1,bbox2)
    area1 = bbox1[2]*bbox1[3]
    area2 = bbox2[2]*bbox2[3]

    max_left = max(bbox1[0]-bbox1[2]/2, bbox2[0]-bbox2[2]/2)
    min_right = min(bbox1[0]+bbox1[2]/2, bbox2[0]+bbox2[2]/2)
    max_top = max(bbox1[1]-bbox1[3]/2, bbox2[1]-bbox2[3]/2)
    min_bottom = min(bbox1[1]+bbox1[3]/2, bbox2[1]+bbox2[3]/2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        intersect = (min_right-max_left)*(min_bottom-max_top)
        return (intersect / area1+area2-intersect)


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, labels):
        batch_size = labels.size()[0]
        # feature_map_size = labels.size()[1] # S
        loss_coord_xy = 0  # coord xy loss
        loss_coord_wh = 0  # coord wh loss
        loss_obj = 0  # obj loss
        loss_noobj = 0  # noobj loss
        loss_class = 0  # classes loss
        for n in range(batch_size):
            for x in range(7):
                for y in range(7):

                    # this region has object
                    if labels[n, x, y, 4] == 1:
                        # calculate iou of two bbox
                        iou1 = calculate_iou(
                            preds[n, x, y, 0:4], labels[n, x, y, 0:4])
                        iou2 = calculate_iou(
                            preds[n, x, y, 5:9], labels[n, x, y, 5:9])
                        # judge responsible box
                        if iou1 > iou2:
                            # calculate coord xy loss
                            loss_coord_xy += 5 * \
                                torch.sum(
                                    (preds[n, x, y, 0:2] - labels[n, x, y, 0:2])**2)

                            # calculate coord wh loss
                            loss_coord_wh += torch.sum(
                                (preds[n, x, y, 2:4].sqrt()-labels[n, x, y, 2:4].sqrt())**2)

                            # calculate obj confidence loss
                            loss_obj += (preds[n, x, y, 4] - iou1)**2

                            # calculate noobj confidence loss
                            loss_noobj += 0.5 * ((preds[n, x, y, 9]-iou2)**2)
                        else:
                            # calculate coord xy loss
                            loss_coord_xy += 5 * \
                                torch.sum(
                                    (preds[n, x, y, 5:7] - labels[n, x, y, 5:7])**2)

                            # calculate coord wh loss
                            loss_coord_wh += torch.sum(
                                (preds[n, x, y, 7:9].sqrt()-labels[n, x, y, 7:9].sqrt())**2)

                            # calculate obj confidence loss
                            loss_obj += (preds[n, x, y, 9] - iou2)**2

                            # calculate noobj confidence loss
                            loss_noobj += 0.5 * ((preds[n, x, y, 4]-iou1)**2)

                        # calculate class loss
                        loss_class += torch.sum(
                            (preds[n, x, y, 10:] - labels[n, x, y, 10:])**2)

                    # this region has no object
                    else:
                        loss_noobj += 0.5 * \
                            torch.sum(preds[n, x, y, [4, 9]]**2)

                # end for y
            # end for x
        # end for batchsize

        # print(loss_coord_xy)
        # print(loss_coord_wh)
        # print(loss_obj)
        # print(loss_noobj)
        # print(loss_class)

        loss = loss_coord_xy + loss_coord_wh + loss_obj + \
            loss_noobj + loss_class  # five loss terms
        return loss/batch_size


def parse_cfg(cfg_path):
    cfg = {}
    with open(cfg_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#' or line == '\n':
                continue
            line = line.strip().split(':')
            key, value = line[0].strip(), line[1].strip()
            cfg[key] = value
    return cfg
