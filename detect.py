import argparse
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from utils import parse_cfg, pred2xywhcc, build_model

parser = argparse.ArgumentParser(description='YOLOv1 Pytorch Implementation')
parser.add_argument("--weights", "-w", default="weights/last.pth", help="Path of model weight", type=str)
parser.add_argument("--source", "-s", default="dataset/VOC2007/JPEGImages",
                    help="Path of your input file source,0 for webcam", type=str)
parser.add_argument('--output', "-o", default='output', help='Output folder', type=str)
parser.add_argument("--cfg", "-c", default="cfg/yolov1.yaml", help="Your model config path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/dataset.yaml", help="Your dataset config path", type=str)
parser.add_argument('--cam_width', "-cw", default=848, help='camera width', type=int)
parser.add_argument('--cam_height', "-ch", default=480, help='camera height', type=int)
parser.add_argument('--conf_thresh', "-ct", default=0.1, help='prediction confidence thresh', type=float)
parser.add_argument('--iou_thresh', "-it", default=0.3, help='prediction iou thresh', type=float)
args = parser.parse_args()

# random colors
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]


def draw_bbox(img, bboxs, class_names):
    h, w = img.shape[0:2]
    n = bboxs.size()[0]
    bboxs = bboxs.detach().numpy()
    print(bboxs)
    for i in range(n):
        p1 = (int((bboxs[i, 0] - bboxs[i, 2] / 2) * w), int((bboxs[i, 1] - bboxs[i, 3] / 2) * h))
        p2 = (int((bboxs[i, 0] + bboxs[i, 2] / 2) * w), int((bboxs[i, 1] + bboxs[i, 3] / 2) * h))
        class_name = class_names[int(bboxs[i, 5])]
        # confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=COLORS[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(bboxs[i, 5])])
    return img


def predict_img(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh):
    """get model prediction of one image

    Args:
        img: image ndarray
        model: pytorch trained model
        input_size: input size
    Returns:
        xywhcc: predict image bbox
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_img = Image.fromarray(img).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    pred_img = transform(pred_img)
    pred_img.unsqueeze_(0)

    pred = model(pred_img)[0].detach().cpu()
    xywhcc = pred2xywhcc(pred, S, B, num_classes, conf_thresh, iou_thresh)

    return xywhcc


if __name__ == "__main__":
    # load configs from config file
    cfg = parse_cfg(args.cfg)
    input_size = cfg['input_size']
    dataset_cfg = parse_cfg(args.dataset_cfg)
    class_names = dataset_cfg['class_names']
    print('Class names:', class_names)
    S, B, num_classes = cfg['S'], cfg['B'], cfg['num_classes']
    conf_thresh, iou_thresh, source = args.conf_thresh, args.iou_thresh, args.source

    # load model
    model = build_model(args.weights, S, B, num_classes)
    print('Model loaded successfully!')

    # create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Image
    if source.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'gif', 'webp']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)

        xywhcc = predict_img(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
        if xywhcc.size()[0] != 0:
            img = draw_bbox(img, xywhcc, class_names)
            # save output img
            cv2.imwrite(os.path.join(args.output, img_name), img)

    # Video
    elif source.split('.')[-1] in ['mp4', 'avi', 'mkv', 'flv', 'rmvb', 'mov', 'rm']:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Video loaded failed!')
                break

            xywhcc = predict_img(frame, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
            if xywhcc.size()[0] != 0:
                frame = draw_bbox(frame, xywhcc, class_names)

            cv2.resizeWindow('frame', int(cap.get(3)), int(cap.get(4)))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Webcam
    elif source == '0':
        cap = cv2.VideoCapture(0)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        # main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Camera loaded failed!')
                break
            print('Frame shape:', frame.shape)

            xywhcc = predict_img(frame, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
            if xywhcc.size()[0] != 0:
                frame = draw_bbox(frame, xywhcc, class_names)

            cv2.resizeWindow('Frame', int(cap.get(3)), int(cap.get(4)))
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Folder
    elif source == source.split('.')[-1]:
        # create output folder
        output = os.path.join(args.output, source.split('/')[-1])
        if os.path.exists(output):
            shutil.rmtree(output)
            # os.removedirs(output)
        os.makedirs(output)

        imgs = os.listdir(source)
        for img_name in imgs:
            # img = cv2.imread(os.path.join(source, img_name))
            img = cv2.imdecode(np.fromfile(os.path.join(
                source, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            # predict
            xywhcc = predict_img(img, model, input_size, S, B, num_classes, conf_thresh, iou_thresh)
            if xywhcc.size()[0] != 0:
                img = draw_bbox(img.copy(), xywhcc, class_names)
                # save output img
                cv2.imwrite(os.path.join(output, img_name), img)
            print(img_name)
