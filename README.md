# yolov1-pytorch
This repo is a pytorch implementation of yolov1.
* Deep Learning based yolov1 object detector, using `Pytorch` deep learning framwork
* This project can be also transplanted to other platforms like `Raspberry Pi`

# Demo
## 1.Run demo directly
cam_demo.py runs to show a UI and recongnize the objects
```bash
$ python cam_demo.py
```

## 2.Run in command line
detect.py runs inference on a variety of sources, cd your project path and type:
```bash
$ python detect.py -c cfg/yolov1.cfg -d cfg/dataset.cfg -w weights/yolov1.pth --source 0  # webcam
                                                                                       file.jpg  # image 
                                                                                       file.mp4  # video
                                                                                       path/*.jpg # img folder path
                                                                                       path/*.mp4 # video folder path
```

# Usage
## Preparation
* 1.Create an empty folder (in this project was `dataset` folder) as your dataset folder
* 2.Prepare your own datasets, you shold move images folder and labels folder into dataset folder, each image must have corresponding same-name `.txt` label in labels folder, each label has `x,y,w,h,c` five values
* 3.Your dataset folder should be like this:
```
dataset/
  ├──images
  |   ├──001.png
  |   ├──002.png
  |   └──003.jpg
  └──labels 
      ├──001.txt
      ├──002.txt
      └──003.txt
```

## Train
* 1.Create an empty model config file `xxx.cfg` (xxx is your project name) in cfg directory (this repo is *cfg/yolov1.cfg*), then imitate `yolov1.cfg` editing customized config. Set **nb_class** according to your class number (this repo nb_class=10).
* 2.Modify **epochs**, **learning rate**, **batch_size** and other hyper parameters in `train.py` depending on actual situations
* 3.Create a empty dataset config file `sss.cfg` (sss is your dataset name) in cfg directory (this repo is *cfg/dataset.cfg*), then imitate `dataset.cfg` editing customized config file.
* 4.Run `train.py` to train your own model (only when dataset was prepared):
```bash 
$ python train.py --cfg cfg/yolov1.cfg --data cfg/dataset.cfg
```
![image](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/data/batch0.png)

## Caution
* Need plotting model structure? Just install `graphviz` and `torchviz` first
* Validation and test periods are among training process, see train.py for more details
* You can also imitate `models/model.py` customizing your own model

# Program Structure Introduction
* cfg: contain some config files
* data: some samples and misc files
* dataset: your own dataset path
* models: some network model structures
* utils: some util and kernel files
* output: output file folder
* weights: model weights

# Requirements
Python 3.X version with all [requirements.txt](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/requirements.txt) dependencies installed, including `torch>=1.2`. To install run:
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
* Nvidia GTX 1060 3GS