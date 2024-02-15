import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
from numpy.random import randint
import numpy as np
import os
import os.path
from utils.logger import logger
import math

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:  #**dataset_conf is the entire JSON "dataset" from .yaml
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")


    def _get_train_indices(self, record, modality='RGB'):
        indices = []    
        if self.dense_sampling[modality]: 
            average_duration = (record.num_frames[modality] - self.num_frames_per_clip[modality] + 1) // self.num_clips
            if average_duration > 0:
                start_indices = np.multiply(list(range(self.num_clips)), average_duration) + randint(average_duration, size=self.num_clips) #if in randint (min, max, size (how many)) if max is None, the first parameter is interpreted as max (NOT INCLUDED)
            else:
                start_indices = np.zeros(self.num_clips)
            
            for start_index in start_indices:
                frame_index = int(start_index)
                for _ in range(self.num_frames_per_clip[modality]):
                    indices.append(frame_index)
                    
                    if (frame_index + self.stride) < record.end_frame:
                        frame_index += self.stride                     
        else: 
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) + np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                indices = np.tile(frame_idx, self.num_clips)
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
        
        return indices
        
        
    def _get_val_indices(self, record, modality):
        indices = []
        if self.dense_sampling[modality]: 
            average_duration = (record.num_frames[modality] - self.num_frames_per_clip[modality] + 1) // self.num_clips
            if average_duration > 0:
                start_indices = np.array([int(average_duration / 2.0 + average_duration * x) for x in range(self.num_clips)])
            else:
                start_indices = np.zeros(self.num_clips)
                
            for start_index in start_indices:
                frame_index = int(start_index)
                for _ in range(self.num_frames_per_clip[modality]):
                    indices.append(frame_index)
                    
                    if (frame_index + self.stride) < record.end_frame:
                        frame_index += self.stride   
        else: 
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) #+ np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                indices = np.tile(frame_idx, self.num_clips)
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
                            
        return indices


    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)




#*###########################
#* ActionSense dataset
#*###########################

#####################################
#####################################
#!preprocessed_data.pkl format:
#!  'index', 'file', 'description_x', 'labels', 'description_y', 'start',
#!       'stop', 'myo_left_timestamps', 'myo_left_readings',
#!       'myo_right_timestamps', 'myo_right_readings'
#!                
#!  
class ActionVisionDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str ->SXY
        modalities: list(str, str, ...) -> ["RGB"] ["EMG"] or ["EMG, "RGB"]
        mode: str ->(train, test)
        dataset_conf must contain the following:  
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, EMG, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info
        
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        #*READ ANNOTATIONS!! Action-Net/data/EMG_datapreprocessed_data
        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        
        #**each record contains its own annotation info (1 line from a file in "train_val/*.pkl") and the whole JSON "dataset" from .yaml (which contains the path where to retrieve the associated video!)
        self.record_list = [ActionSenseRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        #**Load Features if required
        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset_conf[m].features_name +
                                                                          pickle_name)))[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                #!else:
                #!    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            #!self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")


    def _get_train_indices(self, record, modality):
        indices = []    
        if self.dense_sampling[modality]: 
            #*DENSE SAMPLING: equidistant frames (by self.stride) in each clip
            average_duration = (record.num_frames[modality] - self.num_frames_per_clip[modality] + 1) // self.num_clips
            if average_duration > 0:
                start_indices = np.multiply(list(range(self.num_clips)), average_duration) + randint(average_duration, size=self.num_clips) #if in randint (min, max, size (how many)) if max is None, the first parameter is interpreted as max (NOT INCLUDED)
            else:
                start_indices = np.zeros(self.num_clips)
            
            for start_index in start_indices:
                frame_index = int(start_index)
                for _ in range(self.num_frames_per_clip[modality]):
                    indices.append(frame_index)
                    
                    if (frame_index+self.stride < record.end_frame):
                        frame_index += self.stride
                            
        else: 
            #*UNIFORM SAMPLING: equidistant frames
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) + np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                indices = np.tile(frame_idx, self.num_clips)
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
        
        return indices
    
                
        
    def _get_val_indices(self, record, modality):
        indices = []
        if self.dense_sampling[modality]: 
        #*DENSE SAMPLING: equidistant frames (by self.stride) in each clip
            average_duration = (record.num_frames[modality] - self.num_frames_per_clip[modality] + 1) // self.num_clips
            if average_duration > 0:
                start_indices = np.array([int(average_duration / 2.0 + average_duration * x) for x in range(self.num_clips)])
            else:
                start_indices = np.zeros(self.num_clips)
                
            for start_index in start_indices:
                frame_index = int(start_index)
                for _ in range(self.num_frames_per_clip[modality]):
                    indices.append(frame_index)
                    
                    if (frame_index+self.stride < record.end_frame):
                        frame_index += self.stride
        else: 
            #*UNIFORM SAMPLING: equidistant frames
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) #+ np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                indices = np.tile(frame_idx, self.num_clips)
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
                            
        return indices


    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.record_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0] #[720,16]
            if self.additional_info:
                return sample, record.label, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        if self.transform is not None:
            # finally, all the transformations are applied
            process_data = self.transform[modality](images)
        else: 
            process_data = (images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.uid, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.uid,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.uid, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        elif modality == 'EMG':  
            #get the readings (left and right) associated to index selected from the interval
            #stop-start. Since the start->stop interval and the len(myo reading) may be different
            # (not sinchronized by some millisecond): if index overshoots, I take the last reading
            if idx >= len(record.myo_left_readings):
                myo_left_reading = record.myo_left_readings[-1]
            else:
                myo_left_reading = record.myo_left_readings[idx] #(8,)  
                
            if idx >= len(record.myo_right_readings):
                myo_right_readings = record.myo_right_readings[-1] #(8,)
            else:
                myo_right_readings = record.myo_right_readings[idx] #(8,) 
                
            #stack the left-right readings into 1*16
            #if they don't match size, I take intersection!
            combined_myo = np.hstack((myo_right_readings, myo_left_reading))  #(16,) # Horizontally stack the arrays
        
            return [combined_myo]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.record_list)



#class ActionSenseDataset(data.Dataset, ABC):
   
