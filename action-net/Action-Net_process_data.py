import h5py
import numpy as np
import pickle
import re
import pandas as pd
import os
from scipy.signal import butter, lfilter # for filtering


EMG_data_path = 'Action-Net/data/EMG_data'
annotations_path = 'Action-Net/data/annotations'

#!!'S03_1.pkl' TOO LOW FREQUENCY. It's a calibration file?
subjects = ('S00_2.pkl', 'S01_1.pkl', 'S02_2.pkl' , 'S02_3.pkl','S02_4.pkl','S03_2.pkl','S04_1.pkl','S05_2.pkl','S06_1.pkl','S06_2.pkl','S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl')


def lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = np.empty_like(data)
    for i,x in enumerate(data):
        y[i] = lfilter(b, a, x.T).T
    return y


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

    #delete calibration rows that have no myo_right and myo left readings
    combined_df = combined_df.dropna()

    #read annotation files
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))

    #Inner join by indexes and file
    AN_train = annotations_train.merge(combined_df, on=['index','file'], how='inner')
    AN_test = annotations_test.merge(combined_df, on=['index', 'file'], how='inner')

    return AN_train, AN_test


def Baseline(dataset):
    #schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    new_rows = []
    for subjectid in subjects: #SEPARATELY FOR EACH SUBJECT
        indices = np.where(dataset[:, 1] == subjectid)[0]
        data = dataset[indices]
        #internally sort
        indices = np.lexsort((data[:, 0],))
        data = data[indices]
        for i, row in enumerate(data):
            if i < data.shape[0]-1:
                next_action_start_ts = data[i+1][5] #start next
                action_end_ts = data[i][6]
                duration_s = (next_action_start_ts - action_end_ts)
                if duration_s < 5: #!5sec
                    continue

                #! SEARCH IN THIS ACTION'S READINGS AND ALSO IN THE NEXT ONE!!-> doesn't find anything 
                filtered_myo_left_indices = np.where((action_end_ts<= row[7]) & (row[7] < next_action_start_ts))[0]
                filtered_myo_left_ts = np.array([row[7][i] for i in filtered_myo_left_indices])
                filtered_myo_left_readings = np.array([row[8][i] for i in filtered_myo_left_indices])

                filtered_myo_left_indices = np.where((action_end_ts<= data[i+1][7]) & (data[i+1][7] < next_action_start_ts))[0]
                filtered_myo_left_ts = np.array([data[i+1][7][i] for i in filtered_myo_left_indices])
                filtered_myo_left_readings = np.array([data[i+1][8][i] for i in filtered_myo_left_indices])

                if len(filtered_myo_left_indices) == 0:
                    continue

                filtered_myo_right_indices = np.where((action_end_ts <= data[i+1][9]) & (data[i+1][9] < next_action_start_ts))[0]
                filtered_myo_right_ts = np.array([row[9][i] for i in filtered_myo_right_indices])
                filtered_myo_right_readings = np.array([row[10][i] for i in filtered_myo_right_indices])

                if len(filtered_myo_right_indices) == 0:
                    continue

                # Create new rows with the updated arrays
                new_rows.append([row[0], row[1], 'no_activity','baseline', 'no_activity', action_end_ts, next_action_start_ts, filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings])

    #!transform new_rows into a np array
    new_rows = np.array(new_rows, dtype=object)

    if len(new_rows) == 0:
        return dataset

    #Since baseline action will be much more than real action, we keep only 20% of it random instances of it
    random_indices = np.random.choice(len(new_rows), int(len(new_rows)*0.2), replace=False)
    reduced_new_rows = new_rows[random_indices]

    # Stack the augmented rows vertically with the original dataset
    dataset_with_baselines = np.vstack((dataset, reduced_new_rows))

    return dataset_with_baselines


def Augmenting(data):
    #schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']

    # Compute, for all rows, the number of 5-second intervals to split that row into
    new_rows = []
    num_intervals = ((data[:, 6] - data[:, 5]) / 5).astype(int)

    for i, row in enumerate(data):
        if num_intervals[i] <= 1:    #if just 1 interval of 5 seconds, don't split, leave action like this
            new_rows.append(row)
            continue

        # Compute the start and stop timesteps for each interval of this row
        start_ts = row[5] + np.arange(num_intervals[i]) * 5
        stop_ts = start_ts + 5

        for j in range(num_intervals[i]):
            filtered_myo_left_indices = np.where((start_ts[j] <= row[7]) & (row[7] < stop_ts[j]))[0]
            filtered_myo_left_ts = np.array([row[7][i] for i in filtered_myo_left_indices])
            filtered_myo_left_readings = np.array([row[8][i] for i in filtered_myo_left_indices])

            filtered_myo_right_indices = np.where((start_ts[j] <= row[9]) & (row[9] < stop_ts[j]))[0]
            filtered_myo_right_ts = np.array([row[9][i] for i in filtered_myo_right_indices])
            filtered_myo_right_readings = np.array([row[10][i] for i in filtered_myo_right_indices])

            # Create new rows with the updated arrays
            new_row = [row[0], row[1], row[2], row[3], row[4], start_ts[j], stop_ts[j], filtered_myo_left_ts, filtered_myo_left_readings, filtered_myo_right_ts, filtered_myo_right_readings]
            new_rows.append(new_row)

    #!transform new_rows into a np array
    new_rows = np.array(new_rows, dtype=object)

    return new_rows


def Preprocessing(dataset):
    #schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']

    #Define filtering parameter.
    filter_cutoff_Hz = 5
    #absolute + filter + normalize SEPARATELY the myo-right and myo-left readings SEPARATELY for each subject
    for subjectid in subjects: #SEPARATELY FOR EACH SUBJECT
        indices = np.where(dataset[:, 1] == subjectid)[0]
        data = dataset[indices]
        if len(data)==0:
            continue

        #internally sort because of timestamps when computing Fs
        sort_indices = np.lexsort((data[:, 5],))
        data = data[sort_indices]

        myo_left_timestamps =  np.array(data[:,7]) #(299, 0) for S00_2 #array([list([n elements])])
        myo_left_readings = np.array(data[:,8]) # (299,0) for S00_2 array([list([array([8elements])])])
        myo_right_timestamps =  np.array(data[:,9])
        myo_right_readings =  np.array(data[:,10])

        #dataset is divided in actions. For example S00_2 -> 50 rows (actions) each with multiple readings

        myo_left_total_n_readings = np.sum([sub_array.size for sub_array in myo_left_timestamps])
        myo_right_total_n_readings = np.sum([sub_array.size for sub_array in myo_right_timestamps])


        Fs_left = (myo_left_total_n_readings - 1) / (myo_left_timestamps[-1][-1] - myo_left_timestamps[0][0])
        Fs_right = (myo_right_total_n_readings - 1) / (myo_right_timestamps[-1][-1] - myo_right_timestamps[0][0])

        #absolute value
        myo_left_readings = np.abs(myo_left_readings)
        myo_right_readings = np.abs(myo_right_readings)

        #filter
        myo_left_readings = lowpass_filter(myo_left_readings, filter_cutoff_Hz, Fs_left)
        myo_right_readings= lowpass_filter(myo_right_readings, filter_cutoff_Hz, Fs_right)

        #normalize
        reshaped_myo_left_readings = np.concatenate(myo_left_readings, axis=0)
        reshaped_myo_right_readings = np.concatenate(myo_left_readings, axis=0)

        myo_left_readings= (myo_left_readings - np.min(reshaped_myo_left_readings)) / (np.max(reshaped_myo_left_readings)-np.min(reshaped_myo_left_readings)/2)
        myo_right_readings = (myo_right_readings - np.min(reshaped_myo_right_readings)) / (np.max(reshaped_myo_right_readings)-np.min(reshaped_myo_right_readings)/2)

        #shift to [-1,1]
        reshaped_myo_left_readings = np.concatenate(myo_left_readings, axis=0)
        reshaped_myo_right_readings = np.concatenate(myo_right_readings, axis=0)

        myo_left_readings = myo_left_readings - np.min(reshaped_myo_left_readings) -1
        myo_right_readings = myo_right_readings - np.min(reshaped_myo_right_readings) -1

        #*update original
        dataset[indices, 8] = myo_left_readings
        dataset[indices, 10] = myo_right_readings

    return dataset


if __name__ == '__main__':
    
    #Load all EMG data, merge it with annotations
    #split dataset into train and test splits according to annotations files
    AN_train, AN_test = split_train_test() #returns pd dataframe

    ##Convert the datasets to NumPy arrays for better efficiency
    ##schema: ['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    AN_train = AN_train.copy().to_numpy() #[527*11]
    AN_test = AN_test.copy().to_numpy()

    # ##Introduce baselines actions
    # #AN_train_base = Baseline(AN_train) #[*11]
    # #AN_test_base = Baseline(AN_test)   #[*11]

    # #Augment actions into smaller actions of 5seconds each
    AN_train = Augmenting(AN_train)
    AN_test = Augmenting(AN_test)
    
    # #Filter, Normalize and Augment
    AN_train = Preprocessing(AN_train) #AN_train_base #AN_train_aug
    AN_test = Preprocessing(AN_test) #AN_test_base #AN_test_aug
    
    ##Convert back to pd dataframes
    AN_train_final = pd.DataFrame(AN_train, columns=['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings'])
    AN_test_final = pd.DataFrame(AN_test, columns=['index', 'file', 'description_x', 'labels', 'description_y', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings'])

    #twice the column description due to the join, keep one and rename it
    AN_train_final.drop(columns=['description_y'], inplace=True)
    AN_train_final.rename(columns={'description_x': 'description'}, inplace=True)
    AN_test_final.drop(columns=['description_y'], inplace=True)
    AN_test_final.rename(columns={'description_x': 'description'}, inplace=True)

    # #Save preprocessed dataset
    filepath = EMG_data_path + 'SXY_train.pkl'
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(AN_train_final, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    filepath = EMG_data_path + 'SXY_test.pkl'
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(AN_test_final, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    #Produce annotations only for subject S04
    AN_train_final_S04 = AN_train_final[AN_train_final['file'] == 'S04_1.pkl']
    AN_test_final_S04 = AN_test_final[AN_test_final['file'] == 'S04_1.pkl']

    # #Save preprocessed dataset for SO4
    filepath = EMG_data_path + 'S04_1_train.pkl'
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(AN_train_final_S04, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    filepath = EMG_data_path +'S04_1_test.pkl'
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(AN_test_final_S04, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

