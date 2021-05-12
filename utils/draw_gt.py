import os
import random

import cv2

COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]


def draw_gt():
    imgs = os.listdir('../dataset/NWPU VHR-10 dataset/positive image set')
    for img_name in imgs:
        print(img_name)
        img = cv2.imread('../dataset/NWPU VHR-10 dataset/positive image set/' + img_name)
        ori_width, ori_height = img.shape[1], img.shape[0]
        # draw bbox
        with open('../dataset/NWPU VHR-10 dataset/ground truth/' + img_name.split('.')[0] + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    break
                line = line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

                x1, y1, x2, y2, c = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]) - 1

                # x1 y1 x2 y2 normalize to 0~1
                x1, y1, x2, y2 = x1 / ori_width, y1 / ori_height, x2 / ori_width, y2 / ori_height

                # convert to resize 448
                x1, y1, x2, y2 = int(x1 * 448), int(y1 * 448), int(x2 * 448), int(y2 * 448)

                img = cv2.resize(img, (448, 448))

                img = cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c - 1], 2)
                cv2.rectangle(img, (x1 - 1, y1), (x2 + 1, y1 - 22), COLORS[c - 1], -1, cv2.LINE_AA)
                cv2.putText(img, str(c), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(
                    255, 0, 0), thickness=3)
        cv2.imwrite('../dataset/NWPU VHR-10 dataset/gt_image/' + img_name, img)


if __name__ == "__main__":
    draw_gt()
