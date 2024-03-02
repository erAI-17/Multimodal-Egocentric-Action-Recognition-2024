import numpy as np
import pickle
import pandas as pd
import os
from scipy.signal import butter, lfilter # for filtering


EMG_data_path = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/Action-Net-EMG/'
annotations_path = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/annotations/'

subjects = ('S00_2.pkl', 'S01_1.pkl', 'S02_2.pkl' , 'S02_3.pkl','S02_4.pkl', 'S03_1.pkl' ,'S03_2.pkl','S04_1.pkl','S05_2.pkl','S06_1.pkl','S06_2.pkl','S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl')
#subjects = ('S03_1.pkl','S00_2.pkl')


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
    Fs = 160 #!GIVEN
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
    
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[2]):
        channel_serie = data[:,:,channel].flatten()
        filtered_serie = lfilter(b,a, channel_serie)
        filtered_data[:,:,channel] = filtered_serie.reshape(data.shape[0], data.shape[1])
    
    y = lfilter(b, a, data.T).T
    
    # y = np.empty_like(data)
    # for i,x in enumerate(data):
    #     y[i] = lfilter(b, a, x.T).T
      
    return filtered_data


def split_train_test():
    #combine all pkl files from different subjects
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

    #read annotation files
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))

    #Inner join by indexes and file
    AN_train = annotations_train.merge(combined_df, on=['index','file'], how='inner')
    AN_test = annotations_test.merge(combined_df, on=['index', 'file'], how='inner')
    
    #drop additional "description_y" column after the join and rename "description_x" into simple "description"
    AN_train.drop('description_y', axis=1, inplace=True)
    AN_train.rename(columns={'description_x': 'description'}, inplace=True)
    
    AN_test.drop('description_y', axis=1, inplace=True)
    AN_test.rename(columns={'description_x': 'description'}, inplace=True)
    
    return AN_train, AN_test


def Augmenting(data):
    #schema: ['index', 'file', 'description', 'labels', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    augmented_data = []

    for action in data:
        num_intervals = np.ceil((action['stop'] - action['start']) / 5).astype(int)
        
        # Compute the start and stop timesteps for each interval of this action
        start_ts = action['start'] + np.arange(num_intervals) * 5
        stop_ts = start_ts + 5

        n_readings = 750
        for j in range(num_intervals):
            #!myo left augment
            filtered_myo_left_indices = np.where((start_ts[j] <= action['myo_left_timestamps']) & (action['myo_left_timestamps'] < stop_ts[j]))[0]
            
            if filtered_myo_left_indices.shape[0] > n_readings:
                #cut
                filtered_myo_left_indices = filtered_myo_left_indices[:n_readings]
                filtered_myo_left_ts = np.array([action['myo_left_timestamps'][i] for i in filtered_myo_left_indices])
                filtered_myo_left_readings = np.array([action['myo_left_readings'][i] for i in filtered_myo_left_indices]) 
                
            elif filtered_myo_left_indices.shape[0] < n_readings:
                padding_length = n_readings - filtered_myo_left_indices.shape[0]
                filtered_myo_left_ts = np.array([action['myo_left_timestamps'][i] for i in filtered_myo_left_indices])
                filtered_myo_left_readings = np.array([action['myo_left_readings'][i] for i in filtered_myo_left_indices]) 
                #add 0 paddding to readings. Actually, we need also to add entries in the ts array by interpolating new reading timestamps, but we never use that array 
                filtered_myo_left_readings = np.pad(filtered_myo_left_readings, ((0, padding_length), (0,0)), 'constant', constant_values=(0))
            
            #!myo right augment
            filtered_myo_right_indices = np.where((start_ts[j] <= action['myo_right_timestamps']) & (action['myo_right_timestamps'] < stop_ts[j]))[0] 
            
            if filtered_myo_right_indices.shape[0] > n_readings:
                filtered_myo_right_indices = filtered_myo_right_indices[:n_readings]
                filtered_myo_right_ts = np.array([action['myo_right_timestamps'][i] for i in filtered_myo_right_indices])
                filtered_myo_right_readings = np.array([action['myo_right_readings'][i] for i in filtered_myo_right_indices])
            
            elif filtered_myo_right_indices.shape[0] < n_readings:
                padding_length = n_readings - filtered_myo_right_indices.shape[0]
                filtered_myo_right_ts = np.array([action['myo_right_timestamps'][i] for i in filtered_myo_right_indices])
                filtered_myo_right_readings = np.array([action['myo_right_readings'][i] for i in filtered_myo_right_indices])
                #add 0 paddding to readings. Actually, we need also to add entries in the ts array by interpolating new reading timestamps, but we never use that array 
                filtered_myo_right_readings = np.pad(filtered_myo_right_readings, ((0, padding_length), (0,0)), 'constant', constant_values=(0))
                
    
            # Create new action
            new_action = {'index': action['index'],
                          'file': action['file'],
                          'description': action['description'],
                          'labels': action['labels'],
                          'start': start_ts[j],
                          'stop': stop_ts[j],
                          'myo_left_timestamps': filtered_myo_left_ts,
                          'myo_left_readings': filtered_myo_left_readings,
                          'myo_right_timestamps': filtered_myo_right_ts, 
                          'myo_right_readings': filtered_myo_right_readings
                          }
            
            augmented_data.append(new_action)
            
    #!REBALANCE HERE
    
    return augmented_data


def Preprocessing(data):
    #schema: ['index', 'file', 'description', 'labels', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    #Define filtering parameter.
    filter_cutoff_Hz = 5
    #absolute + filter + normalize SEPARATELY the myo-right and myo-left readings SEPARATELY for each subject
    for subjectid in subjects: #SEPARATELY FOR EACH SUBJECT
        subject_data = [action for action in data if action.get('file') == subjectid]
        
        if len(subject_data)  == 0: #in the test split there are no samples for every subject
            continue
        
        #internally sort actions of this subject because of timestamps when computing Fs
        subject_data = sorted(subject_data, key= lambda action: action['start'])
        
        #extract data
        myo_left_timestamps =  np.array([action['myo_left_timestamps'] for action in subject_data]) #(349,750,8)
        myo_left_readings = np.array([action['myo_left_readings'] for action in subject_data]) 
        myo_right_timestamps = np.array([action['myo_right_timestamps'] for action in subject_data]) 
        myo_right_readings = np.array([action['myo_right_readings'] for action in subject_data]) 

        #last action, last timestamp - first action, first timestamp
        Fs_left = ((myo_left_timestamps.shape[1]*myo_left_timestamps.shape[0]) - 1) / (myo_left_timestamps[-1][-1] - myo_left_timestamps[0][0])
        Fs_right = ((myo_right_timestamps.shape[1]*myo_right_timestamps.shape[0]) - 1) / (myo_right_timestamps[-1][-1] - myo_right_timestamps[0][0])

        #absolute value
        myo_left_readings = np.abs(myo_left_readings)
        myo_right_readings = np.abs(myo_right_readings)

        #filter each of 8 channels separately
        myo_left_readings = lowpass_filter(myo_left_readings, filter_cutoff_Hz, Fs_left)
        myo_right_readings= lowpass_filter(myo_right_readings, filter_cutoff_Hz, Fs_right)
        
        #normalize with global max and min
        myo_left_readings= (myo_left_readings) / ((np.max(myo_left_readings)-np.min(myo_left_readings))/2)
        myo_right_readings = (myo_right_readings) / ((np.max(myo_right_readings)-np.min(myo_right_readings))/2)

        #shift to [-1,1] with global min
        myo_left_readings = myo_left_readings - np.min(myo_left_readings) -1
        myo_right_readings = myo_right_readings - np.min(myo_right_readings) -1

        #*update original actions with preprocessed data
        for i, action in enumerate(subject_data):
            action['myo_left_readings'] = myo_left_readings[i]
            action['myo_right_readings'] = myo_right_readings[i]
    
    return data


def Stacking(data):
    for action in data:
        myo_left_readings = action['myo_left_readings']
        myo_right_readings = action['myo_right_readings']
        combined_readings = np.hstack((myo_left_readings, myo_right_readings))
        action['features_EMG'] = combined_readings

    return data

def handler_S04(AN_train_final_df, AN_test_final_df):
    #video_length: 1.01.06 
    fps= 29.67
    
    #Filter lines only for subject S04
    AN_train_final_S04 = AN_train_final_df[AN_train_final_df['file'] == 'S04_1.pkl']
    AN_test_final_S04 = AN_test_final_df[AN_test_final_df['file'] == 'S04_1.pkl']
    
    #Merge back and order by start timestamp
    merged_df = pd.concat([AN_train_final_S04, AN_test_final_S04])
    sorted_merged_df = merged_df.sort_values(by='start', ascending=True)
    
    # Assuming the first timestamp corresponds to the start of the video
    video_start_timestamp = sorted_merged_df['start'].min()

    # Calculate START_INDEX and STOP_INDEX
    sorted_merged_df['start_frame'] = ((sorted_merged_df['start'].astype(float) - float(video_start_timestamp)) * fps).round().astype(int)
    sorted_merged_df['stop_frame'] = ((sorted_merged_df['stop'].astype(float) - float(video_start_timestamp)) * fps).round().astype(int)
    
    #AGAIN split into S04_test and S04_train
    #read annotation files
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))

    #Inner join by indexes and file
    S04_train = annotations_train.merge(sorted_merged_df, on=['index','file'], how='inner')
    S04_test = annotations_test.merge(sorted_merged_df, on=['index', 'file'], how='inner')
    
    #shuffle rows 
    S04_train = S04_train.sample(frac=1).reset_index(drop=True)
    S04_test = S04_test.sample(frac=1).reset_index(drop=True)

    # Save preprocessed dataset for SO4 formatted as uid, subjectid, features_EMG, features_RGB , label
    output_filepath = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/S04_train.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(S04_train, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    output_filepath = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/S04_test.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(S04_test, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    #Load all EMG data and split dataset into train and test splits according to annotations files
    AN_train_df, AN_test_df = split_train_test() #returns pd dataframe
    
    #Convert the datasets to dictionaries with schema: ['index', 'file', 'description', 'labels', 'start', 'stop','myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    AN_train = AN_train_df.to_dict('records') 
    AN_test = AN_test_df.copy().to_dict('records')
    
    #Augment actions into smaller actions of 5seconds each
    AN_train = Augmenting(AN_train) #3821 elements
    AN_test = Augmenting(AN_test) #419
    
    #Filter, Normalize and Augment 
    AN_train = Preprocessing(AN_train) #AN_train_base #AN_train_aug
    AN_test = Preprocessing(AN_test) #AN_test_base #AN_test_aug
    
    #Stack the myo_left_readings and myo_right_readings into a new key "features_EMG" 
    AN_train = Stacking(AN_train) #AN_train_base #AN_train_aug
    AN_test = Stacking(AN_test) #AN_test_base #AN_test_aug
    
    #Convert back to pd dataframes
    AN_train_final_df = pd.DataFrame(AN_train, columns=['index', 'file', 'description', 'labels', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings', 'features_EMG'])
    AN_test_final_df = pd.DataFrame(AN_test, columns=['index', 'file', 'description', 'labels', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings', 'features_EMG'])
    
    #There are some activities with slightly different names that I want to merge 
    activities_renamed = {
        'Open/close a jar of almond butter': ['Open a jar of almond butter'],
        'Get/replace items from refrigerator/cabinets/drawers': ['Get items from refrigerator/cabinets/drawers'],
    }
    
    AN_train_final_df.loc[AN_train_final_df['description'] == 'Open/close a jar of almond butter', 'description'] = 'Open a jar of almond butter'
    AN_test_final_df.loc[AN_test_final_df['description'] == 'Open/close a jar of almond butter', 'description'] = 'Open a jar of almond butter'
    AN_train_final_df.loc[AN_train_final_df['description'] == 'Get/replace items from refrigerator/cabinets/drawers', 'description'] = 'Get items from refrigerator/cabinets/drawers'
    AN_test_final_df.loc[AN_test_final_df['description'] == 'Get/replace items from refrigerator/cabinets/drawers', 'description'] = 'Get items from refrigerator/cabinets/drawers'
    
    # #add class column based on different instances of "description"
    unique_values = AN_train_final_df['description'].unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}
    AN_train_final_df['description_class'] = AN_train_final_df['description'].map(value_to_int)
    AN_test_final_df['description_class'] = AN_test_final_df['description'].map(value_to_int)
    
    #add unique index column identifying each action, because "index" column has the same value for augmented actions
    AN_train_final_df['uid'] = range(len(AN_train_final_df))
    AN_test_final_df['uid'] = range(len(AN_test_final_df))
    
    #shuffle rows 
    AN_train_final_df = AN_train_final_df.sample(frac=1).reset_index(drop=True)
    AN_test_final_df = AN_test_final_df.sample(frac=1).reset_index(drop=True)
    
    #Save preprocessed dataset into pkl file FOR EVERY SUBJECT formatted as {features: [{uid: 1 , subjectid: S00_2, features_EMG: [] , labels: }]}
    output_filepath = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/SXY_train.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(AN_train_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    output_filepath = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/SXY_test.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(AN_test_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    #Produce annotations for S04 also with start and stop frames for RGB flow
    handler_S04(AN_train_final_df, AN_test_final_df)
    