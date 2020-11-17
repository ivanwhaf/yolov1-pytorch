import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
from .util import xywhc2label


class YOLODataset(Dataset):
    def __init__(self, img_path, label_path, transforms=None):
        self.img_path = img_path  # images folder path
        self.label_path = label_path  # labels folder path
        self.transforms = transforms
        self.filenames = os.listdir(img_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.filenames[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image's original weight and height
        ori_width, ori_height = img.shape[1], img.shape[0]

        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)

        xywhc = []
        # read each image's corresponding label .txt
        with open(os.path.join(self.label_path, self.filenames[idx].split('.')[0]+'.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                # process label
                line = line.replace('(', '').replace(')', '').replace(
                    ' ', '').strip().split(',')

                # convert str to int
                x1, y1, x2, y2, c = int(line[0]), int(
                    line[1]), int(line[2]), int(line[3]), int(line[4])

                # x1 y1 x2 y2 --> x y w h
                x, y, w, h = int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1

                # x y w h normalized to 0~1
                x, y, w, h = x / ori_width, y/ori_height, w/ori_width, h/ori_height

                xywhc.append([x, y, w, h, c])

        # xywhc list convert to label
        label = xywhc2label(xywhc)
        label = torch.Tensor(label)
        return img, label


def create_dataloader(img_path, laebl_path, train_proportion, val_proportion, test_proportion, batch_size, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # create yolo dataset
    dataset = YOLODataset(img_path,
                          laebl_path, transforms=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size*train_proportion)
    val_size = int(dataset_size*val_proportion)
    # test_size = int(dataset_size*test_proportion)
    test_size = dataset_size-train_size-val_size

    # split dataset to trainset, valset and testset three parts
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
