import h5py
import numpy as np
import pickle
import re
import os

#Code from https://github.com/delpreto/ActionNet/blob/master/parsing_data/example_parse_hdf5_file.py##

dir = 'C:/Users/39351/Downloads/emg_data/'

#for each hdf5 file in directory dir
for file in os.listdir(dir):
  if file.endswith('.hdf5'):
    file_id = re.search(r'S\d+(_\d+)?', file).group()  #search for patterns 'S' followed by '_' followed by digit and OPTIONALLY followed by _ and another digit 
    filepath1 = dir + 'streamLog_actionNet-wearables_' + file_id + '.hdf5'

    h5_file = h5py.File(filepath1, 'r')

    for i in ('left','right'):
      device_name1 = 'myo-' + i #left #right
      stream_name = 'emg'
      # Get the data as an Nx8 matrix where each row is a timestamp and each column is an EMG channel.
      emg_data = h5_file[device_name1][stream_name]['data']
      emg_data = np.array(emg_data)

      # Get the timestamps for each row as seconds since epoch.
      emg_time_s = h5_file[device_name1][stream_name]['time_s']
      emg_time_s = np.squeeze(np.array(emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list

      # Get the timestamps for each row as human-readable strings.
      emg_time_str = h5_file[device_name1][stream_name]['time_str']
      emg_time_str = np.squeeze(np.array(emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

      ##################################################################################
      device_name2 = 'experiment-activities'
      stream_name = 'activities'
      # Get the timestamped label data.
      # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
      activity_datas = h5_file[device_name2][stream_name]['data']
      activity_times_s = h5_file[device_name2][stream_name]['time_s']
      activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list

      # Convert to strings for convenience.
      activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

      # Combine start/stop rows to single activity entries with start/stop times.
      #   Each row is either the start or stop of the label.
      #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
      exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
      activities_labels = []
      activities_start_times_s = []
      activities_end_times_s = []
      activities_ratings = []
      activities_notes = []
      for (row_index, time_s) in enumerate(activity_times_s):
        label    = activity_datas[row_index][0]
        is_start = activity_datas[row_index][1] == 'Start'
        is_stop  = activity_datas[row_index][1] == 'Stop'
        rating   = activity_datas[row_index][2]
        notes    = activity_datas[row_index][3]
        if exclude_bad_labels and rating in ['Bad', 'Maybe']:
          continue
        # Record the start of a new activity.
        if is_start:
          activities_labels.append(label)
          activities_start_times_s.append(time_s)
          activities_ratings.append(rating)
          activities_notes.append(notes)
        # Record the end of the previous activity.
        if is_stop:
          activities_end_times_s.append(time_s)

      actions = []
      #For each action
      for instance_index, (label, label_start_time_s, label_end_time_s) in enumerate(zip(activities_labels, activities_start_times_s, activities_end_times_s)):
          # Segment the data!
          emg_indexes_forLabel = np.where((emg_time_s >= label_start_time_s) & (emg_time_s <= label_end_time_s))[0] #extract from emg_time_s, all the indexes associated to that action
          emg_data_forLabel = emg_data[emg_indexes_forLabel, :] #data for the action
          emg_time_s_forLabel = emg_time_s[emg_indexes_forLabel] #timestamps for the action
          emg_time_str_forLabel = emg_time_str[emg_indexes_forLabel] #human readable timestamps for the action

          action = {
              'idx': instance_index+1,
              'file': file_id + '.pkl',
              'label': label,
              'start_time_s': label_start_time_s,
              'end_time_s': label_end_time_s,
              'duration_s': label_end_time_s - label_start_time_s,
              'emg_data': emg_data_forLabel,
          }
          
          # Append instance data to the list.
          actions.append(action)


      #Save in pickle file
      filepath2 = './EMG/data' + '_' +file_id + '_' + device_name1 + '.pkl'
      with open(filepath2, 'wb') as pickle_file:
          pickle.dump(actions, pickle_file)
        