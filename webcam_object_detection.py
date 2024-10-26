
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
from object_tracking.observers import ObjectActivationObserver
from object_tracking.applications.counter import ObjectCounter
from graphics.cv.information_drawer import InformationDrawer
from object_tracking.event_manager import ObjectEventManager
import argparse
from datetime import datetime
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./models/mb2-ssd.pth")
parser.add_argument("--label_path", type=str, default="./models/voc-model-labels.txt")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--verbose_detection", type=bool, default=False)

args = parser.parse_args()

model_path = args.model_path
label_path = args.label_path

cap = cv2.VideoCapture(0)
cap.set(3, args.width)
cap.set(4, args.height)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)

net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

timer = Timer()



def _callback_object_in(object_name, object_id, new_object_position):
    print(f"[{datetime.now()}] obj {object_name} - in.")
    object_counter.increase(object_name)

def _callback_object_leave(object_name, object_id, old_object_position):
    print(f"[{datetime.now()}] obj {object_name} leave")

info_drawer = InformationDrawer()
observer = ObjectActivationObserver()
event_manager = ObjectEventManager(_callback_object_in, _callback_object_leave)
object_counter = ObjectCounter()
object_observation_in_last_frame = {}

scale_w = 150.0 / 1920.0
scale_h = 150.0 / 1080.0
hog = cv2.HOGDescriptor()

# frame rendering loops
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, -1, 0.4)
    interval = timer.end()
    if args.verbose_detection:
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        pt_1_1 = int(box[0])
        pt_1_2 = int(box[1])
        pt_2_1 = int(box[2])
        pt_2_2 = int(box[3])
        
        ROI = orig_image[pt_1_2: max(pt_1_2 + 1, pt_2_2), pt_1_1: max(pt_1_1 + 1, pt_2_2)]
        ROI_feature = None
        if not any(x == 0 for x in ROI.shape):
            resized_ROI = cv2.resize(ROI, (150, 150)) 
            ROI_feature = hog.compute(resized_ROI)
        observer.append_object(class_names[labels[i]], pt_1_1, pt_1_2, pt_2_1, pt_2_2, ROI_feature)
        cv2.rectangle(orig_image, (pt_1_1, pt_1_2), (pt_2_1, pt_2_2), (255, 255, 0), 4)
        
        cv2.putText(orig_image, label,
                    (pt_1_1 + 20, pt_1_2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    info_drawer.print_objects_in_current_frame(orig_image, observer, 200, 20)
    info_drawer.print_object_counting_record(orig_image, object_counter, 600, 20)
    event_manager.eval_object_in_event(object_observation_in_last_frame, observer.current_objects)
    event_manager.eval_object_leave_event(object_observation_in_last_frame, observer.current_objects)
    object_observation_in_last_frame = observer.current_objects

    observer.clear()
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()