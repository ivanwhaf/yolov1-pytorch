import argparse
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.util import parse_cfg, pred2xywhc
from models import build_model

# camera shape
CAM_WIDTH, CAM_HEIGHT = 848, 480

# random colors
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]


def draw_bbox(img, bbox, classes):
    h, w = img.shape[0:2]
    n = bbox.size()[0]
    print(bbox)
    for i in range(n):
        p1 = (w*bbox[i, 0], h*bbox[i, 1])
        p2 = (w*bbox[i, 2], h*bbox[i, 3])
        class_name = classes[int(bbox[i, 4])]
        confidence = bbox[i, 5]
        cv2.rectangle(img, p1, p2, color=COLOR[int(bbox[i, 4])])
        cv2.putText(img, class_name, p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return img


def predict_img(img, model, input_size, classes):
    """get model prediction of one image

    Args:
        img: image ndarray
        model: pytorch trained model
    Returns:
        output: pytorch model output
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    img = transform(img)
    img.unsqueeze_(0)

    pred = model(img)
    xywhcc = pred2xywhcc(pred)
    img = draw_bbox(img, xywhcc, classes)
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(
        description='YOLOv1 Pytorch Implementation')

    parser.add_argument("--weight", "-w", dest='weight', default="weights/yolov1.pth",
                        help="Path of model weight", type=str)

    parser.add_argument("--source", "-s", dest='source', default="data/samples/test.jpg",
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
    weight_path, cfg_path, source, output, data = args.weight, args.cfg, args.source, args.output, args.data

    # load configs from config file
    cfg = parse_cfg(cfg_path)
    input_size = int(cfg['input_size'])
    data = parse_cfg(data)
    classes = data['class_names']

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
        img = predict_img(img, model, input_size, classes)
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

            frame = predict_img(frame, model, input_size, classes)
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

            frame = predict_img(frame, model, input_size, classes)
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
            img = predict_img(
                img, model, input_size, classes)
            print(img_name)
            print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
            # save output img
            cv2.imwrite(os.path.join(output, img_name), img)
