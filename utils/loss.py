import torch
from torch import nn

from .util import calculate_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S, B):
        super().__init__()
        self.S = S
        self.B = B

    def forward(self, preds, labels):
        batch_size = labels.size(0)

        loss_coord_xy = 0.  # coord xy loss
        loss_coord_wh = 0.  # coord wh loss
        loss_obj = 0.  # obj loss
        loss_no_obj = 0.  # no obj loss
        loss_class = 0.  # class loss

        for i in range(batch_size):
            for y in range(self.S):
                for x in range(self.S):
                    # this region has object
                    if labels[i, y, x, 4] == 1:
                        # convert x,y to x,y
                        # pred_bbox1 = torch.Tensor(
                        #     [(preds[i, x, y, 0] + x) / 7, (preds[i, x, y, 1] + y) / 7, preds[i, x, y, 2],
                        #      preds[i, x, y, 3]])
                        # pred_bbox2 = torch.Tensor(
                        #     [(preds[i, x, y, 5] + x) / 7, (preds[i, x, y, 6] + y) / 7, preds[i, x, y, 7],
                        #      preds[i, x, y, 8]])
                        # label_bbox = torch.Tensor(
                        #     [(labels[i, x, y, 0] + x) / 7, (labels[i, x, y, 1] + y) / 7, labels[i, x, y, 2],
                        #      labels[i, x, y, 3]])

                        pred_bbox1 = torch.Tensor(
                            [preds[i, y, x, 0], preds[i, y, x, 1], preds[i, y, x, 2], preds[i, y, x, 3]])
                        pred_bbox2 = torch.Tensor(
                            [preds[i, y, x, 5], preds[i, y, x, 6], preds[i, y, x, 7], preds[i, y, x, 8]])
                        label_bbox = torch.Tensor(
                            [labels[i, y, x, 0], labels[i, y, x, 1], labels[i, y, x, 2], labels[i, y, x, 3]])

                        # calculate iou of two bbox
                        iou1 = calculate_iou(pred_bbox1, label_bbox)
                        iou2 = calculate_iou(pred_bbox2, label_bbox)

                        # judge responsible box
                        if iou1 > iou2:
                            # calculate coord xy loss
                            loss_coord_xy += 5 * torch.sum((labels[i, y, x, 0:2] - preds[i, y, x, 0:2]) ** 2)

                            # coord wh loss
                            loss_coord_wh += torch.sum((labels[i, y, x, 2:4].sqrt() - preds[i, y, x, 2:4].sqrt()) ** 2)

                            # obj confidence loss
                            loss_obj += (iou1 - preds[i, y, x, 4]) ** 2
                            # loss_obj += (preds[i, y, x, 4] - 1) ** 2

                            # no obj confidence loss
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 9]) ** 2)
                            # loss_no_obj += 0.5 * ((preds[i, y, x, 9] - 0) ** 2)
                        else:
                            # coord xy loss
                            loss_coord_xy += 5 * torch.sum((labels[i, y, x, 5:7] - preds[i, y, x, 5:7]) ** 2)

                            # coord wh loss
                            loss_coord_wh += torch.sum((labels[i, y, x, 7:9].sqrt() - preds[i, y, x, 7:9].sqrt()) ** 2)

                            # obj confidence loss
                            loss_obj += (iou2 - preds[i, y, x, 9]) ** 2
                            # loss_obj += (preds[i, y, x, 9] - 1) ** 2

                            # no obj confidence loss
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 4]) ** 2)
                            # loss_no_obj += 0.5 * ((preds[i, y, x, 4] - 0) ** 2)

                        # class loss
                        loss_class += torch.sum((labels[i, y, x, 10:] - preds[i, y, x, 10:]) ** 2)

                    # this region has no object
                    else:
                        loss_no_obj += 0.5 * torch.sum((0 - preds[i, y, x, [4, 9]]) ** 2)

                    # end labels have object
                # end for y
            # end for x
        # end for batch size

        # print(loss_coord_xy, loss_coord_wh, loss_obj, loss_no_obj, loss_class)

        loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_no_obj + loss_class  # five loss terms
        return loss / batch_size
