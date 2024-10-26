class ObjectCounter():
    def __init__(self):
        self.counter = {}
    
    def increase(self, object_name):
        if object_name not in self.counter:
            self.counter[object_name] = 1
        else:
            self.counter[object_name] += 1
