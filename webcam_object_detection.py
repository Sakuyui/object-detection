from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mb2-ssd")
parser.add_argument("--model_path", type=str, default="./models/mb2-ssd.pth")
parser.add_argument("--label_path", type=str, default="./models/voc-model-labels.txt")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)

args = parser.parse_args()

net_type = args.model
model_path = args.model_path
label_path = args.label_path


cap = cv2.VideoCapture(0)
cap.set(3, args.width)
cap.set(4, args.height)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
# net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)

net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (int(box[0])+20, int(box[1])+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()