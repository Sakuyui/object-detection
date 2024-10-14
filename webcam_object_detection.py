from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import argparse
from datetime import datetime


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

class ObjectCounter():
    def __init__(self):
        self.counter = {}
    
    def increase(self, object_name):
        if object_name not in self.counter:
            self.counter[object_name] = 1
        else:
            self.counter[object_name] += 1

class ObjectActivationObserver():
    def __init__(self):
        self.current_objects = {
        }

    def append_object(self, label, pos_x1, pos_y1, pos_x2, pos_y2):
        if label in self.current_objects:
            self.current_objects[label]['count'] += 1
            self.current_objects[label]['positions'].append([pos_x1, pos_y1, pos_x2, pos_y2])
        else:
            self.current_objects[label] = {
                'count': 1,
                'positions': [[pos_x1, pos_y1, pos_x2, pos_y2]]
            }
    
    def objects(self):
        for key in self.current_objects:
            yield (key, self.current_objects[key])
            
    def clear(self):
        self.current_objects = {}
    
class InformationDrawer():
    def __init__(self):
        pass
    def print_objects_in_current_frame(self, frame, observer: ObjectActivationObserver, x_1, y_1):
        current_x = x_1
        current_y = y_1
        for object_label, record in observer.objects():
            cv2.putText(frame, f'{object_label}: {record["count"]}',
                    (current_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (0, 0, 255),
                    2)  # line type
            y_1 += 20
    
    def print_object_counting_record(self, frame, object_counter: ObjectCounter, x_1, y_1):
        current_x = x_1
        current_y = y_1
        for object_name in object_counter.counter:
            cv2.putText(frame, f'{object_name}: {object_counter.counter[object_name]}',
                    (current_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (0, 0, 255),
                    2)  # line type
            y_1 += 20
    
class ObjectEventManager():
    def __init__(self, callback_object_in, callback_object_leave):
        self.callback_object_in = callback_object_in
        self.callback_object_leave = callback_object_leave
    
    def eval_object_in_event(self, last_frame_object_observation, current_frame_object_observation):
        for object_name in current_frame_object_observation:
            current_record = current_frame_object_observation[object_name]
            tackle_as_new_object = True
            for current_position_record in current_record['positions']:
                for last_position_record in last_frame_object_observation.get(object_name,{}).get('positions', []):
                    last_x1 = last_position_record[0]
                    last_y1 = last_position_record[1]
                    last_x2 = last_position_record[2]
                    last_y2 = last_position_record[3]

                    current_x1 = current_position_record[0]
                    current_y1 = current_position_record[1]
                    current_x2 = current_position_record[2]
                    current_y2 = current_position_record[3]

                    # Calculate the boundaries of the overlap region
                    _left = max(current_x1, last_x1)
                    _right = min(current_x2, last_x2)
                    _top = max(current_y1, last_y1)
                    _bottom = min(current_y2, last_y2)

                    # Check if the rectangles overlap
                    if (_bottom - _top <= 0 or _right - _left <= 0):
                        overlap_area = 0.0
                    else:
                        # Calculate the area of the overlapping region
                        overlap_area = (_bottom - _top) * (_right - _left)

                    # If you need the overlap percentage, you could compute it based on one of the rectangles
                    # For example:
                    last_area = (last_x2 - last_x1) * (last_y2 - last_y1)
                    overlap_percentage = overlap_area / last_area if last_area > 0 else 0.0
                    if overlap_percentage > 0.5:
                        tackle_as_new_object = False
                        break
                if tackle_as_new_object:
                    self.callback_object_in(object_name, current_position_record)

    def eval_object_leave_event(self, last_frame_object_observation, current_frame_object_observation):
        for object_name in last_frame_object_observation:
            last_record = last_frame_object_observation[object_name]
            tackle_as_new_object = False
            for last_position_record in last_record['positions']:
                for current_position_record in current_frame_object_observation.get(object_name, {}).get('positions', []):
                    last_x1 = last_position_record[0]
                    last_y1 = last_position_record[1]
                    last_x2 = last_position_record[2]
                    last_y2 = last_position_record[3]

                    current_x1 = current_position_record[0]
                    current_y1 = current_position_record[1]
                    current_x2 = current_position_record[2]
                    current_y2 = current_position_record[3]

                    # Calculate the boundaries of the overlap region
                    _left = max(current_x1, last_x1)
                    _right = min(current_x2, last_x2)
                    _top = max(current_y1, last_y1)
                    _bottom = min(current_y2, last_y2)

                    # Check if the rectangles overlap
                    if (_bottom - _top <= 0 or _right - _left <= 0):
                        overlap_area = 0.0
                    else:
                        # Calculate the area of the overlapping region
                        overlap_area = (_bottom - _top) * (_right - _left)

                    # If you need the overlap percentage, you could compute it based on one of the rectangles
                    # For example:
                    last_area = (last_x2 - last_x1) * (last_y2 - last_y1)
                    overlap_percentage = overlap_area / last_area if last_area > 0 else 0.0
                    if overlap_percentage > 0.5:
                        tackle_as_new_object = True
                        break
                if not tackle_as_new_object:
                    self.callback_object_leave(object_name, last_position_record)
                    
def _callback_object_in(object_name, new_object_position):
    print(f"[{datetime.now()}] obj {object_name} - in")
    object_counter.increase(object_name)

def _callback_object_leave(object_name, old_object_position):
    print(f"[{datetime.now()}] obj {object_name} - leave")

info_drawer = InformationDrawer()
observer = ObjectActivationObserver()
event_manager = ObjectEventManager(_callback_object_in, _callback_object_leave)
object_counter = ObjectCounter()
object_observation_in_last_frame = {}

# opencv loops
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
        observer.append_object(class_names[labels[i]], pt_1_1, pt_1_2, pt_2_1, pt_2_2)
        cv2.rectangle(orig_image, (pt_1_1, pt_1_2), (pt_2_1, pt_2_2), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (pt_1_1 + 20, pt_1_2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    info_drawer.print_objects_in_current_frame(orig_image, observer, 200, 20)
    info_drawer.print_object_counting_record(orig_image, object_counter, 400, 20)
    event_manager.eval_object_in_event(object_observation_in_last_frame, observer.current_objects)
    event_manager.eval_object_leave_event(object_observation_in_last_frame, observer.current_objects)
    object_observation_in_last_frame = observer.current_objects

    observer.clear()
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()