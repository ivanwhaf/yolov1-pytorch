import torch
from torch import nn


def calculate_iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1,bbox2=bbox1.cpu().detach().numpy().tolist(),bbox2.cpu().detach().numpy().tolist()
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
                        # judge responsable box
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
