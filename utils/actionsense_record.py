from .video_record import VideoRecord
class ActionSenseRecord(VideoRecord):
    '''
    tup: a line from one of pkl files in annotations (train_val/...)
    dataset_conf: whole JSON "dataset" from .yaml
    '''
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf
    
    @property
    def uid(self):
        return self._series['uid']
    
    @property
    def subjectid(self):
        return self._series['file'] #S04_0.pkl...

    @property
    def start_frame(self):
        return int(self._series['start_frame'] - 1)

    @property
    def end_frame(self):
        return int(self._series['stop_frame'] - 2)


    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'EMG': self.end_frame - self.start_frame}

    @property
    def label(self):
        if 'description_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['description_class']
    
    @property
    def features_EMG(self):
        return self._series['features_EMG']
    
    @property
    def myo_left_readings(self):
        return self._series['myo_left_readings']
    
    @property
    def myo_right_readings(self):
        return self._series['myo_right_readings']
    
    @property
    def myo_left_ts(self):
        return self._series['myo_left_timestamps']
    
    @property
    def myo_right_ts(self):
        return self._series['myo_right_timestamps']
    
