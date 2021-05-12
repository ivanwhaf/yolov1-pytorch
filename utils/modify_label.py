import os

import cv2

labels_path = 'dataset/NWPU VHR-10 dataset/ground truth/'
images_path = 'dataset/NWPU VHR-10 dataset/positive image set/'
labels = os.listdir(labels_path)

for label in labels:
    image_name = label.split('.')[0]
    print(image_name)
    img = cv2.imread(os.path.join(images_path, image_name + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_width, ori_height = img.shape[1], img.shape[0]  # image's original width and height

    with open(os.path.join(labels_path, label), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            # process label
            line = line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

            # convert str to int
            x1, y1, x2, y2, c = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]) - 1

            # x1 y1 x2 y2 --> x y w h
            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

            # x y w h normalized to 0~1
            x, y, w, h = x / ori_width, y / ori_height, w / ori_width, h / ori_height

            print(x, y, w, h, c)

            f2 = open('dataset/NWPU VHR-10 dataset/gt/' + label, 'a')
            f2.write("{} {} {} {} {}\n".format(str(round(x, 8)), str(round(y, 8)), str(round(w, 8)), str(round(h, 8)),
                                               str(c)))
            f2.close()
