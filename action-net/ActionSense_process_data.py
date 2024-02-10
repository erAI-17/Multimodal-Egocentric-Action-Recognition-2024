import h5py
import numpy as np
import pickle
import re
import pandas as pd
import os
from scipy.signal import butter, lfilter # for filtering


EMG_data_path = 'Action-Net/data/EMG_data'
annotations_path = 'Action-Net/data/annotations'

#! Specify the labels to include.
baseline_label = 'None'
activities_to_classify = [
  baseline_label,
  'Get/replace items from refrigerator/cabinets/drawers',
  'Peel a cucumber',
  'Clear cutting board',
  'Slice a cucumber',
  'Peel a potato',
  'Slice a potato',
  'Slice bread',
  'Spread almond butter on a bread slice',
  'Spread jelly on a bread slice',
  'Open/close a jar of almond butter',
  'Pour water from a pitcher into a glass',
  'Clean a plate with a sponge',
  'Clean a plate with a towel',
  'Clean a pan with a sponge',
  'Clean a pan with a towel',
  'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Stack on table: 3 each large/small plates, bowls',
  'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  ]
baseline_index = activities_to_classify.index(baseline_label)
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
  'Open/close a jar of almond butter': ['Open a jar of almond butter'],
  'Get/replace items from refrigerator/cabinets/drawers': ['Get items from refrigerator/cabinets/drawers'],
}

# Define the modalities to use.
# Each entry is (device_name, stream_name, extraction_function)
# where extraction_function can select a subset of the stream columns.
device_streams_for_features = [
  ('myo-left', 'emg', lambda data: data),
  ('myo-right', 'emg', lambda data: data), 
]

def combine_pickle_files(output_file):
        combined_df = pd.DataFrame() 

        for file_name in os.listdir(EMG_data_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(EMG_data_path, file_name)
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    content['file'] = file_name
                    content['index']  = np.arange(content.shape[0])
                    if isinstance(content, pd.DataFrame):
                        combined_df = pd.concat([combined_df, content], ignore_index=True)
        
        with open(output_file, 'wb') as out:
            pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)


def lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs 
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data.T).T
    return y


def split_train_test():
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))
  
    emg_dataset = pd.read_pickle(os.path.join(EMG_data_path, 'combined_emg_dataset.pkl'))
    AN_train = annotations_train.merge(emg_dataset, on=['index','file'], how='inner')
    AN_test = annotations_test.merge(emg_dataset, on=['index', 'file'], how='inner')
    
    return AN_train, AN_test


def Baseline(dataset):
    new_rows = []
    for i, row in enumerate(dataset): 
        if i < dataset.shape[0]-1:
            noActivity_start_ts = dataset[i+1][5]
            noActivity_end_ts = dataset[i][6]
            duration_s = (noActivity_start_ts - noActivity_end_ts )/6000
            if duration_s < 5: #!5sec
                continue

            filtered_myo_left_ts = [ts for ts in row[7] if noActivity_start_ts <= ts < noActivity_end_ts]
            filtered_myo_left_readings = [row[8][i] for i, ts in enumerate(row[7]) if np.any(((noActivity_start_ts <= ts) & (ts <= noActivity_end_ts)))]
            
            filtered_myo_right_ts = [ts for ts in row[9] if noActivity_start_ts <= ts < noActivity_end_ts]
            filtered_myo_right_readings = [row[10][i] for i, ts in enumerate(row[9]) if np.any(((noActivity_start_ts <= ts) & (ts <= noActivity_end_ts)))]

            new_rows.append([row[0], row[1], 'no_activity','baseline', 'no_activity', noActivity_end_ts, noActivity_start_ts, filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings])
          
    dataset_with_baselines = np.vstack((dataset, np.array(new_rows, dtype=object)))
    
    return dataset_with_baselines
    

def Augmenting(dataset):
    # Compute, for all rows, the number of 5-second intervals to split that row into
    num_intervals = int((dataset[:, 4] - dataset[:, 3]) / 5) #.astype(int)
    
    new_rows = []
    for i, row in enumerate(dataset): 
        
        if num_intervals == 0:
            continue
        
        start_ts = row[3] + np.arange(num_intervals[i]) * 5
        stop_ts = start_ts + 5

        filtered_myo_left_ts = [ts for ts in row[5] if start_ts <= ts < stop_ts]
        filtered_myo_left_readings = [row[6][i] for i, ts in enumerate(row[5]) if np.any(((start_ts <= ts) & (ts <= stop_ts)))]
        
        filtered_myo_right_ts = [ts for ts in row[7] if start_ts <= ts < stop_ts]
        filtered_myo_right_readings = [row[8][i] for i, ts in enumerate(row[7]) if np.any(((start_ts <= ts) & (ts <= stop_ts)))]

        new_rows.append(np.column_stack((start_ts, stop_ts, filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings)))

    augmented_df = np.vstack((dataset, np.concatenate(new_rows)))
  
    return augmented_df


def Preprocessing(dataset):
    filter_cutoff_Hz = 5

    t = dataset['time_s']
    for myo_key in ('myo_left_readings', 'myo_right_readings'):
        
        Fs = (t.size - 1) / (t[-1] - t[0])      
        
        print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (myo_key, Fs, filter_cutoff_Hz))
        y = np.abs(dataset[myo_key])
        y = lowpass_filter(y, filter_cutoff_Hz, Fs)

        print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (myo_key, np.amin(y), np.amax(y)))
        y = y / ((np.amax(y) - np.amin(y))/2) 
        y = y - np.amin(y) - 1
        
        dataset[myo_key] = y
    
    return dataset


if __name__ == '__main__':
    
    #Combine all pkl data files from different subjects into one
    # combine_pickle_files(os.path.join(EMG_data_path ,'combined_emg_dataset.pkl'))
  
    #Load all EMG data, merge it with annotations 
    #split dataset into train and test splits according to annotations files
    AN_train, AN_test = split_train_test()
    
    print(AN_train.columns)
    print(AN_train.shape)
    print(AN_train.head())
    
    # Convert the datasets to NumPy arrays for better efficiency
    AN_train = AN_train.copy().to_numpy() 
    AN_test = AN_test.copy().to_numpy()
    
    #Introduce baselines actions
    AN_train_base = Baseline(AN_train) #[735*11] 
    AN_test_base = Baseline(AN_test)   #[*11]
     
    #Augment actions into smaller actions of 5seconds each
    # AN_train_aug = Augmenting(AN_train_base) 
    # AN_test_aug = Augmenting(AN_test_base) 

    # #Filter, Normalize and Augment
    # AN_train_aug_preprocessed = Preprocessing(AN_train_aug) 
    # AN_test_aug_preprocessed = Preprocessing(AN_test_aug) 
    
    #Convert back to pd dataframes
    # #Save preprocessed dataset
    # filepath = './ActionSense/data/' + 'preprocessed_data' + '.pkl'
    # with open(filepath, 'wb') as pickle_file:
    #     pickle.dump(df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open(filepath, 'wb') as pickle_file:
    #     pickle.dump(df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
