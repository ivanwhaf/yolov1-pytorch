import os
import random
import argparse
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.util import parse_cfg, pred2xywhcc
from models import build_model

# camera shape
CAM_WIDTH, CAM_HEIGHT = 848, 480

# random colors
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]


def draw_bbox(img, bboxs, classes):
    h, w = img.shape[0:2]
    n = bboxs.size()[0]
    print(bboxs)
    for i in range(n):
        p1 = ((bboxs[i, 0]-bboxs[i, 2]/2)*w, (bboxs[i, 1]-bboxs[i, 3]/2)*h)
        p2 = ((bboxs[i, 0]+bboxs[i, 2]/2)*w, (bboxs[i, 1]+bboxs[i, 3]/2)*h)
        class_name = classes[int(bboxs[i, 5])]
        confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=COLORS[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(bboxs[i, 5])])
    return img


def predict_img(img, model, input_size):
    """get model prediction of one image

    Args:
        img: image ndarray
        model: pytorch trained model
        input_size: input size
        classes: classes
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

    pred = model(pred_img)[0]
    xywhcc = pred2xywhcc(pred)

    return xywhcc


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(
        description='YOLOv1 Pytorch Implementation')

    parser.add_argument("--weights", "-w", dest='weights', default="weights/last.pth",
                        help="Path of model weight", type=str)

    parser.add_argument("--source", "-s", dest='source', default="dataset/NWPU VHR-10 dataset/positive image set",
                        help="Path of your input file source,0 for webcam", type=str)

    parser.add_argument('--output', "-o", dest='output', default='output',
                        help='Output folder', type=str)

    parser.add_argument("--cfg", "-c", dest='cfg', default="cfg/yolov1.cfg",
                        help="Your model config path", type=str)

    parser.add_argument("--data", "-d", dest='data', default="cfg/dataset.cfg",
                        help="Your dataset config path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    weight_path, cfg_path, source, output, data = args.weights, args.cfg, args.source, args.output, args.data

    # load configs from config file
    cfg = parse_cfg(cfg_path)
    input_size = int(cfg['input_size'])
    data = parse_cfg(data)
    classes = data['class_names'].split(',')
    print('classes:', classes)

    # laod model
    model = build_model(weight_path, cfg)
    print('Model successfully loaded!')

    # create output folder
    if not os.path.exists(output):
        os.makedirs(output)

    # image
    if source.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'gif']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)

        xywhcc = predict_img(img, model, input_size)
        if xywhcc.size()[0] != 0:
            img = draw_bbox(img, xywhcc, classes)
            # save output img
            cv2.imwrite(os.path.join(output, img_name), img)

    # video
    elif source.split('.')[-1] in ['mp4', 'avi', 'mkv', 'flv', 'rmvb', 'mov', 'rm']:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Video load failed!')
                break

            xywhcc = predict_img(frame, model, input_size)
            if xywhcc.size()[0] != 0:
                frame = draw_bbox(frame, xywhcc, classes)

            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # webcam
    elif source == '0':
        cap = cv2.VideoCapture(0)
        cap.set(3, CAM_WIDTH)
        cap.set(4, CAM_HEIGHT)
        # main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Camera frame load failed!')
                break
            print('frame shape:', frame.shape)

            xywhcc = predict_img(frame, model, input_size)
            if xywhcc.size()[0] != 0:
                frame = draw_bbox(frame, xywhcc, classes)

            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # folder
    elif source == source.split('.')[-1]:
        imgs = os.listdir(source)
        for img_name in imgs:
            # img = cv2.imread(os.path.join(source, img_name))
            img = cv2.imdecode(np.fromfile(os.path.join(
                source, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)

            # create output folder
            output = os.path.join(output, source.split('/')[-1])
            if os.path.exists(output):
                os.remove(output)
            os.makedirs(output)

            # predict
            xywhcc = predict_img(img, model, input_size)
            if xywhcc.size()[0] != 0:
                img = draw_bbox(img, xywhcc, classes)
                # save output img
                cv2.imwrite(os.path.join(output, img_name), img)

            print(img_name)
            # print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
            # save output img
            cv2.imwrite(os.path.join(output, img_name), img)
