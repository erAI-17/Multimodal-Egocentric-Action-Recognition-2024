import numpy as np
import pickle
import pandas as pd
import os
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering

EMG_data_path = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/Action-Net-EMG/'
annotations_path = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/annotations/'

subjects = ('S00_2.pkl', 'S01_1.pkl', 'S02_2.pkl' , 'S02_3.pkl','S02_4.pkl', 'S03_1.pkl' ,'S03_2.pkl','S04_1.pkl','S05_2.pkl','S06_1.pkl','S06_2.pkl','S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl')

# Define segmentation parameters.
resampled_Fs = 10 # define a resampling rate
num_segments_per_action = 20  # 20
segment_duration_s = 5
buffer_startActivity_s = 2
buffer_endActivity_s = 2

def lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    y = lfilter(b, a, data.T).T
      
    return y


def split_train_test():
    #combine all pkl files from different subjects
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store the merged data

    for file_name in os.listdir(EMG_data_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(EMG_data_path, file_name)
            with open(file_path, 'rb') as f:
                content = pd.read_pickle(f)
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


def Preprocessing(data, flag = ''):
    #Define filtering parameter.
    filter_cutoff_Hz = 5
      
    for subjectid in subjects: #SEPARATELY FOR EACH SUBJECT
        subject_data = [action for action in data if action.get('file') == subjectid]
        
        if len(subject_data)  == 0: #in the test split there are no samples for every subject
            continue
    
        subject_data = sorted(subject_data, key= lambda action: action['start'])
        
        for action in subject_data:
            for key in ['myo_right', 'myo_left']:
                #extract data
                action_data =  np.array(action[key +'_readings'])
                ts = np.array(action[key +'_timestamps'])

                Fs= ((ts.size) - 1) / (ts[-1] - ts[0])
            
                action_data = np.abs(action_data)

                y =  lowpass_filter(action_data, filter_cutoff_Hz, Fs)
                
                y = y / ((np.amax(y) - np.amin(y))/2)
                y = y - np.amin(y) - 1

                target_time_s = np.linspace(ts[0], ts[-1],
                                                num=int(round(1+resampled_Fs*(ts[-1] - ts[0]))),
                                                endpoint=True)
                fn_interpolate = interpolate.interp1d(
                        ts, # x values
                        y,   # y values
                        axis=0,              # axis of the data along which to interpolate
                        kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                        fill_value='extrapolate' # how to handle x values outside the original range
                    )

                action_data_resampled = fn_interpolate(target_time_s)
                if np.any(np.isnan(action_data_resampled)):
                        print('\n'*5)
                        print('='*50)
                        print('='*50)
                        print('FOUND NAN')
                        timesteps_have_nan = np.any(np.isnan(action_data_resampled), axis=tuple(np.arange(1,np.ndim(action_data_resampled))))
                        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
                        print('\n'*5)
                        action_data_resampled[np.isnan(action_data_resampled)] = 0
                     
                #*update original actions with preprocessed data
                action[key + '_readings'] = action_data_resampled
                action[key + '_timestamps'] = target_time_s
    return data


def Augmenting(data):
    augmented_data = []
    for action in data:
        
        # Compute the start and stop timesteps for each interval of this action
        start_ts = action['start'] 
        stop_ts = action['stop'] 
        duration_s = stop_ts - start_ts
        if duration_s < segment_duration_s:
            continue

        segment_start_times_s = np.linspace(start_ts, stop_ts - segment_duration_s,
                                            num = num_segments_per_action,
                                            endpoint=True)
        
        keep_action = True
        for _, segment_start_time_s in enumerate(segment_start_times_s):
            
            segment_end_time_s = segment_start_time_s + segment_duration_s
            
            combined_readings = np.empty(shape=(resampled_Fs * segment_duration_s, 0))
            
            for key in ['myo_right', 'myo_left']:
                
                filtered_myo_indices = np.where((segment_start_time_s <= action[key + '_timestamps']) & (action[key + '_timestamps'] < segment_end_time_s))[0]
                
                filtered_myo_indices = list(filtered_myo_indices)
                while len(filtered_myo_indices) < segment_duration_s*resampled_Fs:
                    if filtered_myo_indices[0] > 0: # != 0
                        filtered_myo_indices = [filtered_myo_indices[0]-1] + filtered_myo_indices
                    elif filtered_myo_indices[-1] < len(action[key + '_timestamps'])-1:
                        filtered_myo_indices.append(filtered_myo_indices[-1]+1)
                    else: 
                        keep_action = False
                        break
                      
                while len(filtered_myo_indices) > segment_duration_s*resampled_Fs:
                    filtered_myo_indices.pop()
                    
                filtered_myo_indices = np.array(filtered_myo_indices)
    
                if keep_action:            
                    filtered_myo_key_readings = np.array([action[key + '_readings'][i] for i in filtered_myo_indices]) 

                    combined_readings = np.concatenate((combined_readings, filtered_myo_key_readings), axis=1)
            
            if keep_action:          
                 #! Create new action. For start and stop I use the myo_left timestamps
                new_action = {  'index': action['index'],
                                'file': action['file'],
                                'description': action['description'],
                                'labels': action['labels'],
                                'start': segment_start_time_s,
                                'stop': segment_end_time_s,
                                'features_EMG': combined_readings,
                                }

                keep_action = True
                augmented_data.append(new_action)
            
    return augmented_data



def handler_S04(df, flag):
    fps= 30
    
    S04_df = df[df['file'] == 'S04_1.pkl']
    
    # Assuming the first timestamp corresponds to the start of the video
    # 1655239123.020082 from myo-left corresponds to 22:38: same day
    # 1655217500.0 at 16:38:20
    # 1655219928.0 at 17:18:48
    
    video_start_timestamp =  1655239123.020082 #sorted_merged_df['start'].min() #16552-39974.420555

    S04_df = S04_df.copy()
    S04_df['start_frame'] = ((S04_df['start'] - video_start_timestamp) * fps).astype(int).copy()
    S04_df['stop_frame'] = ((S04_df['stop'] - video_start_timestamp) * fps).astype(int).copy()

    # Save preprocessed dataset for SO4 formatted as uid, subjectid, features_EMG, features_RGB , label
    filepath = '/content/drive/MyDrive/AML/AML_Project_2024/data/Action-Net/S04_'+flag+'.pkl'
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(S04_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    #Load all EMG data and split dataset into train and test splits according to annotations files
    AN_train_df, AN_test_df = split_train_test() #returns pd dataframe
    
    #Convert the datasets to dictionaries with schema: ['index', 'file', 'description', 'labels', 'start', 'stop','myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    AN_train = AN_train_df.to_dict('records') 
    AN_test = AN_test_df.copy().to_dict('records')
    
    #Filter, Normalize
    AN_train = Preprocessing(AN_train, flag='train')
    AN_test = Preprocessing(AN_test, flag='test')
    
    #Augment actions into smaller actions 
    AN_train = Augmenting(AN_train) 
    AN_test = Augmenting(AN_test)
    
    #Convert back to pd dataframes
    AN_train_final_df = pd.DataFrame(AN_train, columns=['index', 'file', 'description', 'labels', 'start','stop','features_EMG'])
    AN_test_final_df = pd.DataFrame(AN_test, columns=['index', 'file', 'description', 'labels', 'start','stop', 'features_EMG'])

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
    handler_S04(AN_train_final_df, flag = 'train')
    handler_S04(AN_test_final_df, flag = 'test')
    
