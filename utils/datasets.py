import os

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .util import xywhc2label


class YOLODataset(Dataset):
    def __init__(self, img_path, label_path, S, B, num_classes, transforms=None):
        self.img_path = img_path  # images folder path
        self.label_path = label_path  # labels folder path
        self.transforms = transforms
        self.filenames = os.listdir(img_path)
        self.filenames.sort()
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read image
        img = cv2.imread(os.path.join(self.img_path, self.filenames[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ori_width, ori_height = img.shape[1], img.shape[0]  # image's original width and height

        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)  # resize and to tensor

        # read each image's corresponding label(.txt)
        xywhc = []
        with open(os.path.join(self.label_path, self.filenames[idx].split('.')[0] + '.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split(' ')

                # convert xywh str to float, class str to int
                x, y, w, h, c = float(line[0]), float(line[1]), float(line[2]), float(line[3]), int(line[4])

                xywhc.append((x, y, w, h, c))

        label = xywhc2label(xywhc, self.S, self.B, self.num_classes)  # convert xywhc list to label
        label = torch.Tensor(label)
        return img, label


def create_dataloader(img_path, label_path, train_proportion, val_proportion, test_proportion, batch_size, input_size,
                      S, B, num_classes):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # create yolo dataset
    dataset = YOLODataset(img_path, label_path, S, B, num_classes, transforms=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_proportion)
    val_size = int(dataset_size * val_proportion)
    test_size = int(dataset_size * test_proportion)
    # test_size = dataset_size - train_size - val_size

    # split dataset to train set, val set and test set three parts
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
