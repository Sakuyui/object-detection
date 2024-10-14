global last_allocated_object_id
last_allocated_object_id = 0
class ObjectActivationObserver():
    def __init__(self):
        self.current_objects = {
        }

    def append_object(self, object_name, pos_x1, pos_y1, pos_x2, pos_y2, feature):
        global last_allocated_object_id
        last_allocated_object_id += 1
        if object_name in self.current_objects:
            self.current_objects[object_name]['count'] += 1
            self.current_objects[object_name]['records'][last_allocated_object_id] = {
                'id': last_allocated_object_id,
                'position': [pos_x1, pos_y1, pos_x2, pos_y2],
                'features': feature
            }

        else:
            self.current_objects[object_name] = {
                'count': 1,
                'records': {
                    last_allocated_object_id: {
                        'id':last_allocated_object_id,
                        'position': [pos_x1, pos_y1, pos_x2, pos_y2],
                        'features': feature
                    }
                }
            }
    
    def objects(self):
        for key in self.current_objects:
            yield (key, self.current_objects[key])
            
    def clear(self):
        self.current_objects = {}