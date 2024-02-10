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
        combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store the merged data

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
    '''
    Function that applies a low-pass filter to a given data set allowing you to remove high-frequency noise or unwanted frequency components from the data 
    by filtering input data removing the frequencies above the cutoff frequency. 
    Cutoff filters are useful for:
        1) noise reduction: remove high-frequency components that may contain unwanted noise or interference
        2) frequency isolation: isolate specific frequency bands of interest. For example, in audio engineering, a low-pass filter can be used to isolate the bass frequencies
        3) Anti-aliasing
        4) Smoothing: smooth out a signal by removing high-frequency fluctuations (unwanted variations) and highlight underlying trends

    cutoff: maximum frequency allowed to pass through the filter.
    Fs:sampling frequency of the input data (it is the number of samples taken per second) 
    order (optional): the order of the filter. The default value is 5, which indicates a 5th-order filter. 
    '''  

    # Nyquist frequency represents the highest frequency that can be accurately represented in the sampled data (set to half of the sampling frequency).
    nyq = 0.5 * Fs 
    #This normalization step ensures that the cutoff frequency is expressed as a fraction of the Nyquist frequency, which is a standard practice in signal processing. 
    normal_cutoff = cutoff / nyq 
    
    #butter  function from the  scipy.signal  module to design the filter coefficients (b,a) for the low-pass filter. 
    #The 'butter' function takes the filter order, the normalized cutoff frequency, and the filter type as arguments. 
    #In this case, the filter type is specified as 'low', indicating a low-pass filter, and the 
    #analog  parameter is set to  False , indicating that we are designing a digital filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    #filter coefficients (b,a) are then used to apply the low-pass filter to the input data. 
    #This is done using the  lfilter  function from the  scipy.signal  module. 
    #The  lfilter  function takes the filter coefficients, the input data, and the axis along which the filter should be applied as arguments. 
    #In this case, the filter is applied along the columns of the input data, as indicated by the  axis=1  argument. 
    y = lfilter(b, a, data.T).T
    
    return y


def split_train_test():
    #read annotation files
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))
    emg_dataset = pd.read_pickle(os.path.join(EMG_data_path, 'combined_emg_dataset.pkl'))

    #Inner join by indexes and file
    AN_train = annotations_train.merge(emg_dataset, on=['index','file'], how='inner')
    AN_test = annotations_test.merge(emg_dataset, on=['index', 'file'], how='inner')
    
    return AN_train, AN_test


def Baseline(dataset):
    #schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    new_rows = []
    for i, row in enumerate(dataset): 
        if i < dataset.shape[0]-1:
            noActivity_start_ts = dataset[i+1][5]
            noActivity_end_ts = dataset[i][6]
            duration_s = (noActivity_start_ts - noActivity_end_ts )/6000
            if duration_s < 5: #!5sec
                continue

            # Filter the readings timestamps and data arrays to keep only the values within the current interval
            filtered_myo_left_ts = [ts for ts in row[7] if noActivity_start_ts <= ts < noActivity_end_ts]
            filtered_myo_left_readings = [row[8][i] for i, ts in enumerate(row[7]) if np.any(((noActivity_start_ts <= ts) & (ts <= noActivity_end_ts)))]
            
            filtered_myo_right_ts = [ts for ts in row[9] if noActivity_start_ts <= ts < noActivity_end_ts]
            filtered_myo_right_readings = [row[10][i] for i, ts in enumerate(row[9]) if np.any(((noActivity_start_ts <= ts) & (ts <= noActivity_end_ts)))]

            # Create new rows with the updated arrays
            new_rows.append([row[0], row[1], 'no_activity','baseline', 'no_activity', noActivity_end_ts, noActivity_start_ts, filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings])
          
    # Stack the augmented rows vertically with the original dataset
    dataset_with_baselines = np.vstack((dataset, np.array(new_rows, dtype=object)))
    
    return dataset_with_baselines
    

def Augmenting(dataset):
    # Compute, for all rows, the number of 5-second intervals to split that row into
    num_intervals = int((dataset[:, 4] - dataset[:, 3]) / 5) #.astype(int)
    
    new_rows = []
    for i, row in enumerate(dataset): 
        
        if num_intervals == 0:
            continue
        
        # Compute the start and stop timesteps for each interval of this row
        start_ts = row[3] + np.arange(num_intervals[i]) * 5
        stop_ts = start_ts + 5

        # Filter the readings timestamps and data arrays to keep only the values within the current interval
        filtered_myo_left_ts = [ts for ts in row[5] if start_ts <= ts < stop_ts]
        filtered_myo_left_readings = [row[6][i] for i, ts in enumerate(row[5]) if np.any(((start_ts <= ts) & (ts <= stop_ts)))]
        
        filtered_myo_right_ts = [ts for ts in row[7] if start_ts <= ts < stop_ts]
        filtered_myo_right_readings = [row[8][i] for i, ts in enumerate(row[7]) if np.any(((start_ts <= ts) & (ts <= stop_ts)))]

        # Create new rows with the updated arrays
        new_rows.append(np.column_stack((start_ts, stop_ts, filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings)))

    # Stack the augmented rows vertically with the original dataset
    augmented_df = np.vstack((dataset, np.concatenate(new_rows)))

    # Convert the NumPy array back to a Pandas dataframe
    augmented_df = pd.DataFrame(augmented_df, columns=dataset.columns)

    return augmented_df


def Preprocessing(dataset):
    #Define filtering parameter.
    filter_cutoff_Hz = 5

    #absolute + filter + normalize SEPARATELY the myo-right and myo-left readings
    t = dataset['time_s']
    for myo_key in ('myo_left_readings', 'myo_right_readings'):
        
        #!Define sampling rate: num_samples/(max-min)
        Fs = (t.size - 1) / (t[-1] - t[0])      
        
        print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (myo_key, Fs, filter_cutoff_Hz))
        y = np.abs(dataset[myo_key])
        y = lowpass_filter(y, filter_cutoff_Hz, Fs)

        print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (myo_key, np.amin(y), np.amax(y)))
        # Normalize 8 dimensions jointly.
        y = y / ((np.amax(y) - np.amin(y))/2) #!why /2 ??????
        # Shift the minimum to -1 instead of 0.
        y = y - np.amin(y) - 1
        
        dataset[myo_key] = y
    
    return dataset


if __name__ == '__main__':
    
    #Combine all pkl data files from different subjects into one
    # combine_pickle_files(os.path.join(EMG_data_path ,'combined_emg_dataset.pkl'))
    
    # df = pd.read_pickle('./Action-Net/data/EMG_data/combined_emg_dataset.pkl')
    # print(df.columns)
    # print(df.shape)
    # print(df.head())
    
    #Load all EMG data, merge it with annotations 
    #split dataset into train and test splits according to annotations files
    AN_train, AN_test = split_train_test()
    
    print(AN_train.columns)
    print(AN_train.shape)
    print(AN_train.head())
    
    # Convert the datasets to NumPy arrays for better efficiency
    #schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    AN_train = AN_train.copy().to_numpy() #[527*11]
    AN_test = AN_test.copy().to_numpy() #[529, 11]
    
    #Introduce baselines actions!
    AN_train_base = Baseline(AN_train) #[735*11] 
    AN_test_base = Baseline(AN_test)   #[*11]
    
    #!TRY PRINT
    # AN_train_base_DF = pd.DataFrame(AN_train_base, columns=['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings'])
    # print()
    # print()
    # print(AN_train_base_DF.columns)
    # print(AN_train_base_DF.shape)
    # print(AN_train_base_DF.head())
    # print(AN_train_base_DF.query('labels == "baseline"'))
    
    
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
    