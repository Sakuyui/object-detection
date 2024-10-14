from object_tracking.observers import ObjectActivationObserver
from object_tracking.applications.counter import ObjectCounter
import cv2

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
            current_y += 20
    
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
            current_y += 20