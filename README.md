# yolov1-pytorch

This repo is a pytorch implementation of yolov1.

* Deep Learning based yolov1 object detector, using `Pytorch` deep learning framework
* This project can be also transplanted to other edge platforms like `Raspberry Pi`
* Paper: [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640v5.pdf)

# Demo

## Run in command line

detect.py runs inference on a variety of sources, cd your project path and type:

```bash
$ python detect.py -c cfg/yolov1.cfg -d cfg/dataset.cfg -w weights/yolov1.pth --source 0  # webcam
                                                                                       file.jpg  # image 
                                                                                       file.mp4  # video
                                                                                       path/*.jpg # img folder path
                                                                                       path/*.mp4 # video folder path
```

![image](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/data/test_demo.jpg)

# Usage

## Preparation

* 1.Create an empty folder (in this project was `dataset` folder) as your dataset folder
* 2.Prepare your own datasets, you should move images folder and labels folder into dataset folder. Each image must have
  corresponding same-name `.txt` label in labels folder. Each label file has several rows, each row represents an
  object, each object has `x,y,w,h,c` five values, which represents `center_x, center_y, width, height, class`. The
  coord value should be normalized to `0~1`
  ![image](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/data/xywh.jpg)
* 3.Your dataset folder should be like this:

```
dataset/{name}/
  ├──images
  |   ├──001.png
  |   ├──002.png
  |   └──003.jpg
  └──labels 
      ├──001.txt
      ├──002.txt
      └──003.txt
```

* 4.Your label `.txt` file should be like this:

```
0.253262 0.252352 0.135673 0.524523 0
0.467335 0.662623 0.265423 0.113139 1
....
```

## Train

* 1.Edit `cfg/yolov1.cfg` config file, and set **num_classes** according to class number of dataset (this repo
  num_classes=10)
* 2.Edit `cfg/dataset.cfg` config file, and set **class_names** as class names, **images** as dataset image path, **
  labels** as dataset label path
* 3.Modify **epochs**, **learning rate**, **batch_size** and other hyper parameters in `train.py` depending on actual
  situations
* 4.Run `train.py` to train your own yolov1 model:

```bash 
$ python train.py --cfg cfg/yolov1.cfg --data cfg/dataset.cfg
```

![image](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/data/batch0.png)

## Caution

* Need plotting model structure? Just install `graphviz` and `torchviz` first
* Validation and test periods are among training process, see train.py for more details
* You can also imitate `models/model.py` customizing your own model

# Program Structure Introduction

* cfg: some yolo and dataset config files
* data: some samples and demo jpgs
* dataset: your own dataset path
* models: some network model structure files
* utils: some util and kernel files
* output: output file folder
* weights: model weights

# Requirements

Python 3.X version with all [requirements.txt](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/requirements.txt)
dependencies installed, including `torch>=1.2`. To install run:

```bash
$ pip install -r requirements.txt
```

# Environment

## PC Ⅰ

* Ubuntu 20.04
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* torch 1.2.0
* Nvidia MX350 2G

## PC Ⅱ

* Windows 10
* Python 3.6.8
* CUDA 10.2
* cuDNN 7.6
* torch 1.6.0
* Nvidia GTX 1060 3G

## PC Ⅲ

* Windows 10
* Python 3.8.6
* CUDA 10.1
* cuDNN 7.6.5
* torch 1.7.0
* Nvidia GTX 1080 TI 11G