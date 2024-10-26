import numpy as np

class ObjectEventManager():
    def __init__(self, callback_object_in, callback_object_leave):
        self.callback_object_in = callback_object_in
        self.callback_object_leave = callback_object_leave
    
    def eval_object_in_event(self, last_frame_object_observation, current_frame_object_observation):
        for object_name in current_frame_object_observation:
            current_record = current_frame_object_observation.get(object_name, {}).get('records', {})
            
            for index, current_record_key in enumerate(current_record):
                tackle_as_new_object = True
                last_record = last_frame_object_observation.get(object_name,{}).get("records", {})
                current_position_record = current_record[current_record_key]['position']
                for last_record_key in last_record:
                    last_position_record = last_record[last_record_key]['position']
                    
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
                        last_feature = last_record[last_record_key]['features']
                        current_feature = current_record[current_record_key]['features']
                        if last_feature is not None and current_feature is not None:
                            feature_similarity = np.dot(current_feature, last_feature) / (np.linalg.norm(current_feature) * np.linalg.norm(last_feature))
                            if feature_similarity < 0.55:
                                print(" - feature similarity:", feature_similarity)
                                tackle_as_new_object = True
                        break
                if tackle_as_new_object:
                    self.callback_object_in(object_name, current_record_key, current_position_record)
        
    def eval_object_leave_event(self, last_frame_object_observation, current_frame_object_observation):
        for object_name in last_frame_object_observation:
            # object record at the last frame
            last_record = last_frame_object_observation.get(object_name, {}).get("records", {})
            # iterate each object of a certain category.
            for index, last_record_key in enumerate(last_record):
                tackle_as_new_object = False
                last_position_record = last_record[last_record_key]['position']

                current_record = current_frame_object_observation.get(object_name, {}).get('records', {})
                # find a well-match object in current frame's object record.
                for current_record_key in current_record:
                    current_position_record = current_record[current_record_key]['position']
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
                    # if two boxes are well overlapped and feature is so similarity
                    if overlap_percentage > 0.7:
                        tackle_as_new_object = True
                        # last_feature = last_record['features'][index]
                        # current_feature = current_record['features'][index]
                        # feature_similarity = np.dot(current_feature, last_feature) / (np.linalg.norm(current_feature) * np.linalg.norm(last_feature))
                        # if feature_similarity < 0.7:
                        #     print(" - feature similarity:", feature_similarity)
                        #     tackle_as_new_object = False
                        break
                if not tackle_as_new_object:
                    self.callback_object_leave(object_name, last_record_key, last_position_record)
                    