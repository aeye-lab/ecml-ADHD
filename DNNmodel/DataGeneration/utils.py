import numpy as np
import os
import pandas as pd
import csv
import json
from scipy.io import loadmat
from tqdm import tqdm
import re

def deg2pix(deg, screenPX,screenCM,distanceCM):
  # Converts pixel screen coordinate to degrees of visual angle
  # screenPX is the number of pixels that the monitor has in the horizontal
  # axis (for x coord) or vertical axis (for y coord)
  # screenCM is the width/height of the monitor in centimeters
  # distanceCM is the distance of the monitor to the retina
  # pix: screen coordinate in pixels
  # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
  deg=np.array(deg)
  # center screen coordinates such that 0 is center of the screen:
  #if adjust_origin:
      #pix = pix-(screenPX-1)/2 # pixel coordinates start with (0,0)
  # eye-to-screen-distance in pixels
  distancePX = distanceCM*(screenPX/screenCM)
  pix = np.tan(deg * np.pi/180) * distancePX
  #deg = np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad
  #if reverse_axis:
      #deg=-deg
  return pix

def load_all_sub_info(video_name, sub_info_path):
    file_path = sub_info_path + f'/sub_sel_all.csv'
    df = pd.read_csv(file_path, sep='\t')
    df = df[df[f'video_{video_name}'] == True]
    sub_list = df.Patient_ID.values.tolist()
    return df, sub_list

def load_px(px_load_path):
    #load X files
    X_tmp = np.load(px_load_path, allow_pickle=True)
    X_tmp = pd.DataFrame(X_tmp, columns = ['video_frame', 'Time','X_right_orig', 'Y_right_orig'])
    #only consider the right eye, and the time period of watching the video for moment
    X_tmp=X_tmp.iloc[:,:4]
    X_tmp = X_tmp[X_tmp.video_frame>=0]
    X_tmp.reset_index(drop=True, inplace=True)
    X = X_tmp.copy()
    return X

def pix2deg(pix, screenPX,screenCM,distanceCM, adjust_origin=True, reverse_axis=False):
  # Converts pixel screen coordinate to degrees of visual angle
  # screenPX is the number of pixels that the monitor has in the horizontal
  # axis (for x coord) or vertical axis (for y coord)
  # screenCM is the width/height of the monitor in centimeters
  # distanceCM is the distance of the monitor to the retina
  # pix: screen coordinate in pixels
  # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
  pix=np.array(pix)
  # center screen coordinates such that 0 is center of the screen:
  if adjust_origin:
      pix = pix-(screenPX-1)/2 # pixel coordinates start with (0,0)
  # eye-to-screen-distance in pixels
  distancePX = distanceCM*(screenPX/screenCM)
  deg = np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad
  if reverse_axis:
      deg=-deg
  return deg


def create_extrac_mask(point, x_width, dispsize=(800,600)):
    # compute gaussian matrix
    #gaussian center
    gs_c = x_width/2
    #gaussian std
    gs_st = x_width/6 #--SD: 1.5 deg
    # matrix of zeros
    gaus = np.zeros([x_width,x_width],dtype=float)
    # gaussian matrix
    for i in range(x_width):
        for j in range(x_width):
            gaus[j,i] = np.exp(-1.0 * (((float(i)-gs_c)**2/(2*gs_st*gs_st)) + ((float(j)-gs_c)**2/(2*gs_st*gs_st))))

    #make extraction mask
    strt_x = int(x_width/2)
    strt_y = int(x_width/2)
    EM_size = int(dispsize[1] + 2*strt_y), int(dispsize[0] + 2*strt_x)
    EM = np.zeros(EM_size, dtype=float)

    x = int(strt_x + int(point[0]) - int(x_width/2))
    y = int(strt_y + int(point[1]) - int(x_width/2))

    # add Gaussian to the current matrix
    EM[y:int(y+x_width),x:int(x+x_width)] += gaus
    # resize EM
    EM = EM[strt_y:int(dispsize[1]+strt_y),strt_x:int(dispsize[0]+strt_x)]
    # remove outliers (four corners)
    lowbound = np.mean(EM[EM>0])
    EM[EM<lowbound] = 0
    #normalized by the sum
    EM = EM/np.sum(EM)
    return EM

###################################################################
#
# dispersion algorithm
#
###################################################################

'''
Eye movements were processed with the biometric
framework described in Section 2, with eye movement
classification thresholds: velocity threshold of 20°/sec,
micro-saccade threshold of 0.5°, and micro-fixation
threshold of 100 milliseconds. Feature extraction was
performed across all eye movement recordings, while
matching and information fusion were performed
according to the methods described in Section 3"

source: https://www.researchgate.net/publication/220811146_Identifying_fixations_and_saccades_in_eye-tracking_protocols

The I-DT algorithm requires two parameters, the dispersionthreshold
and the duration threshold.  Like the velocitythreshold for I-VT,
 the dispersion threshold can be set toinclude 1/2° to 1° of visual
 angle if the distance from eye toscreen is known.  Otherwise, the
 dispersion threshold can beestimated from exploratory analysis of
 the data.  The durationthreshold is typically set to a value between
 100 and 200 ms[21], depending on task processing demands.

Identifying fixations and saccades in eye-tracking protocols
Dario Salvucci, H. Goldberg

'''
#
# input:
#           x_coordinates: degrees of visual angle in x-axis
#           y_coordinates: degrees of visual angle in y-axis
#
# output:
#           d: data-frame containing the saccade label
def get_i_dt(x_coordinates,y_coordinates,
            corrupt = None,
            sampling = 1000,
            min_duration = 80,
            velocity_threshold = 20,
            min_event_duration_fixation = 50,
            min_event_duration_saccade = 10,
            flag_skipNaNs = True,
            verbose=0):

    duration_threshold = int(np.floor((min_duration / 1000.) * sampling))
    #min_duration_threshold_fixation = np.max([1,int(np.floor((min_event_duration_fixation / 1000.) * sampling))])
    #min_duration_threshold_saccade = np.max([1,int(np.floor((min_event_duration_saccade / 1000.) * sampling))])
    dispersion_threshold = (velocity_threshold / 1000. * min_duration)

    d = { 'x_deg':x_coordinates,
          'y_deg':y_coordinates}
    d = pd.DataFrame(d)

    sacc = np.ones([len(x_coordinates),])
    start_id = 0
    end_id   = start_id + duration_threshold
    previous_dispension = 100 * dispersion_threshold
    counter = 0
    while start_id <= len(x_coordinates):
        cur_x_window = x_coordinates[start_id:end_id]
        cur_y_window = y_coordinates[start_id:end_id]
        # skip NaNs
        if flag_skipNaNs:
            cur_use_ids = np.logical_and(np.isnan(cur_x_window) == False,
                                        np.isnan(cur_y_window) == False)
            cur_x_window = cur_x_window[cur_use_ids]
            cur_y_window = cur_y_window[cur_use_ids]
        else:
            cur_use_ids = np.logical_and(np.isnan(cur_x_window),
                                        np.isnan(cur_y_window))
            cur_x_window[cur_use_ids] = 100 * dispersion_threshold
            cur_y_window[cur_use_ids] = 100 * dispersion_threshold
        if len(cur_x_window) > 0:
            cur_dispersion = (np.max(cur_x_window) - np.min(cur_x_window)) +\
                            (np.max(cur_y_window) - np.min(cur_y_window))
        else:
            cur_dispersion = 100* dispersion_threshold
        #print('x_coordintes: ' + str(cur_x_window))
        #print('y_coordintes: ' + str(cur_y_window))
        #print('cur_dispersion: ' + str(cur_dispersion))

        if cur_dispersion <= dispersion_threshold and end_id <= len(x_coordinates):
            end_id += 1
            #print('start_id: ' + str(start_id))
            #print('end_id: ' + str(end_id))
            #print(allo)
        else:
            if previous_dispension <= dispersion_threshold:
                sacc[start_id:end_id-1] = 0
                start_id = end_id
                end_id = start_id + duration_threshold
            else:
                start_id += 1
                end_id += 1
        previous_dispension = cur_dispersion
        counter += 1
        if verbose:
            if counter % 1000 == 0:
                print(counter)
    sacc[np.isnan(d['x_deg'])] = 3

    d['sac'] = sacc

    if corrupt is None:
        nan_ids_x = list(np.where(np.isnan(x_coordinates))[0])
        nan_ids_y = list(np.where(np.isnan(y_coordinates))[0])
        corruptIdx = list(set(nan_ids_x + nan_ids_y))
    else:
        corruptIdx = list(np.where(corrupt == 1)[0])
        nan_ids_x = list(np.where(np.isnan(x_coordinates))[0])
        nan_ids_y = list(np.where(np.isnan(y_coordinates))[0])
        corruptIdx = list(set(corruptIdx + nan_ids_x + nan_ids_y))

    # corrupt samples
    d['corrupt'] = d.index.isin(corruptIdx)
    d['event'] = np.where(d.corrupt, 3, np.where(d.sac,2,1))
    return d


# get list of saccade-ids and fixation-ids with dispersion algorithm
def get_sacc_fix_lists_dispersion(x_deg, y_deg,
                        corrupt = None,
                        sampling_rate = 1000,
                        min_duration = 80,
                        velocity_threshold = 20,
                        min_event_duration_fixation = 50,
                        min_event_duration_saccade = 10,
                        flag_skipNaNs = True,
                        verbose=0):

    #  fix=1, saccade=2, corrupt=3
    events = get_i_dt(x_deg, y_deg,
                        corrupt = corrupt,
                        sampling = sampling_rate,
                        min_duration = min_duration,
                        velocity_threshold = velocity_threshold,
                        min_event_duration_fixation = min_event_duration_fixation,
                        min_event_duration_saccade = min_event_duration_saccade,
                        flag_skipNaNs = flag_skipNaNs,
                        verbose=0)

    #  fix=1, saccade=2, corrupt=3
    event_list = np.array(events['event'])
    prev_label = -1
    fixations = []
    saccades = []
    errors = []
    for i in range(len(event_list)):
        cur_label = event_list[i]
        if cur_label != prev_label:
            if prev_label != -1:
                if prev_label == 1:
                    fixations.append(cur_list)
                elif prev_label == 2:
                    saccades.append(cur_list)
                else:
                    errors.append(cur_list)
            cur_list = [i]
        else:
            cur_list.append(i)
        prev_label = cur_label
    if len(cur_list) > 0:
        if prev_label == 1:
            fixations.append(cur_list)
        elif prev_label == 2:
            saccades.append(cur_list)
        else:
            errors.append(cur_list)
    return {'fixations': fixations,
            'saccades': saccades,
            'errors': errors}, events



def get_event_list(X,Y, screenPX_x=800,
                           screenPX_y=600,
                           screenCM_x=33.8,
                           screenCM_y=27.0,
                           distanceCM=63.5,
                           min_saccade_duration = 2):


    x_pixel = X
    y_pixel = Y

    gaze_points = np.concatenate((x_pixel.reshape(-1,1), y_pixel.reshape(-1,1)), axis=1)
    gaze_points[np.all(gaze_points == 0, axis=1),:] = np.nan
    gaze_points[np.any(gaze_points < 0, axis=1),:] = np.nan
    gaze_points[gaze_points[:, 0] >=800,:] = np.nan
    gaze_points[gaze_points[:,1] >= 600,:] = np.nan
    corrupt = np.isnan(gaze_points[:,0])

    x_dva = pix2deg(x_pixel, screenPX_x, screenCM_x, distanceCM, adjust_origin=True)
    y_dva = pix2deg(y_pixel, screenPX_y, screenCM_y, distanceCM, adjust_origin=True)

    #  fix=1, saccade=2, corrupt=3
    list_dicts, event_df = get_sacc_fix_lists_dispersion(x_dva, y_dva,
                                              corrupt = corrupt, sampling_rate = 120)

    return list_dicts, event_df


def load_HBN_sub_info(hbn_data_path):
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'BasicData.csv'))
    df_basic = df_basic[df_basic.Participant_Status == 'Complete']

    return df_basic

def load_sampling_rate(df, hbn_data_path):
    f = open(os.path.join(hbn_data_path, 'hz_overview.json'))
    sampling_rate_file = f.read()
    f.close()
    user_sampling_rate = json.loads(sampling_rate_file)
    df['sampling_rate'] = ""
    for sub in df.Patient_ID.tolist():
        if user_sampling_rate.get(sub) is None:#sampling rate = 30, exclude them
            df.loc[df.Patient_ID == sub, 'sampling_rate'] = np.nan
        else:
            df.loc[df.Patient_ID == sub, 'sampling_rate'] = user_sampling_rate.get(sub)
    return df

def get_video_trigger_time(mat_data):
    """
    compute trigger information (video start time, end time, time difference and video index) of EM recordings
    arguments
    f_path		-	full path to an *.mat file over which the eyemovement measures are recorded
    """
    start_time = np.nan
    end_time = np.nan
    message_data = mat_data['messages'][0]
    start_sig='Message: 8'
    end_sig = 'Message: 10'
    for i in range(len(message_data)):
        cur_line = message_data[i][0]
        if start_sig in cur_line: # start messages may occur twice, the second one seems more reasonable, the time will be overwriten.
            start_time = int(cur_line.split('\t')[0])
            video_indx = int(cur_line.split('\r')[0][-1])
        if end_sig in cur_line:
            end_time = int(cur_line.split('\t')[0])
    if np.isnan(start_time) or np.isnan(end_time):
        time_difference = np.nan
    else:
        time_difference = end_time - start_time
    #print('time difference:', time_difference)
    return start_time, end_time, time_difference, video_indx

def test_time_difference(time_difference, video_indx):
    if video_indx ==1:
        in_range = (117377876 <= time_difference <= 117427994)
    elif video_indx ==2:
        in_range = (162977233 <= time_difference <= 163076389)
    elif video_indx ==3:
        in_range = (170523436 <= time_difference <= 170581836)
    elif video_indx ==4:
        in_range = (203054577 <= time_difference <= 203092531)
    return in_range

def get_sampling_rate(mat_data):
    comment_data = mat_data['comments'][0]
    for i in range(len(comment_data)):
        cur_line = comment_data[i][0]
        if 'Sample Rate:' in cur_line:
            sample_rate = int(cur_line.split('\t')[-1])
            return sample_rate
    return None

def check_tracker_loss_prop(mat_data, threshold, start_point, end_point):
    col_names = dict()
    for idx, col in enumerate(mat_data['colheader'][0]):
        col_names[col[0]] = idx

    eye_data = mat_data['data'][np.logical_and((mat_data['data'][:, col_names['Time']] <= end_point),(mat_data['data'][:, col_names['Time']] >= start_point))][:, [col_names['Time'], col_names['L POR X [px]'], col_names['L POR Y [px]'], col_names['R POR X [px]'], col_names['R POR Y [px]']]]
    #left = eye_data[:,1:3]
    right = eye_data[:,[0,3,4]]

    sr = get_sampling_rate(mat_data)
    X = pd.DataFrame(right.copy())
    X.columns = ['Time','X_right_orig', 'Y_right_orig']
    # time is in microseconds -> transform to ms
    X['Time'] = X['Time'] / 1000.

    #calculate sampling interval between two data points
    ts = X.Time.values
    ts_diff = ts[1:] - ts[:-1]
    ts_diff=np.pad(ts_diff, (0,1), 'constant')#last point add 0
    X['ts_diff'] = ts_diff

    #Check if the sampling rate was recorded incorrectly, if error, return in_rage False
    sr_error_flag=False
    if sr == 120:
        sr_interval = 8.4
        if np.median(ts_diff) > 12:
            in_range = False
            return in_range
    elif sr == 60:
        sr_interval = 16.66
        if np.median(ts_diff) > 20:
            in_range = False
            return in_range

    else:#false error rate
        in_range = False
        return in_range

    #calculate the number of missing points for the missing periods
    num_miss_points = np.round(X['ts_diff'].values/sr_interval) - 1
    miss_count_right = np.sum(num_miss_points[:-1])
    right_points = np.concatenate((X.X_right_orig.values.reshape(-1,1), X.Y_right_orig.values.reshape(-1,1)), axis=1)
    zeros_count_right = np.sum(np.logical_and(right_points[:,0]==0, right_points[:,1]==0))
    tl_prop = (miss_count_right + zeros_count_right)/(len(X)+miss_count_right)

    if tl_prop > threshold:
        in_range = False
    else:
        in_range = True
    return in_range


def count_subject_for_video_task(users, threshold_tl, hbn_data_path):
    #params
    video_1_sub, video_2_sub, video_3_sub, video_4_sub = [], [], [], []
    hbn_et_data_path = hbn_data_path + 'EEG-ET/et_mats/data/'
    for cur_person in tqdm(users):
        cur_load_dir = hbn_et_data_path + cur_person + '/'
        try:
            person_files = os.listdir(cur_load_dir)
        except FileNotFoundError:
            continue

        pat_video = r'{}_Video[1-4]_bothEyes_ET.mat'.format(cur_person)
        pat_video2 = r'{}_Video-[A-Z][A-Z]_bothEyes_ET.mat'.format(cur_person)
        person_video_files = [f for f in person_files if re.match(pat_video, f) or re.match(pat_video2, f)]
        #print(len(person_video_files))
        if not person_video_files:
            continue
        for j in range(len(person_video_files)):
            cur_load_path = cur_load_dir + person_video_files[j]
            mat_data = loadmat(cur_load_path)
            start_point, end_point, time_difference, video_indx = get_video_trigger_time(mat_data)

            #sanity check
            if video_indx not in [1,2,3,4]: #video index marker was wrong
                continue

            if np.isnan(time_difference): # start time or end time missing
                continue

            # data duration must equal video duration
            in_range = test_time_difference(time_difference, video_indx)
            if not in_range:
                print(time_difference, video_indx)
                continue

            #the tracker loss must be less then 10%
            in_range = check_tracker_loss_prop(mat_data, threshold_tl, start_point, end_point)
            if not in_range:
                continue

            if video_indx == 1:
                video_1_sub.append(cur_person)
            elif video_indx == 2:
                video_2_sub.append(cur_person)
            elif video_indx == 3:
                video_3_sub.append(cur_person)
            elif video_indx == 4:
                video_4_sub.append(cur_person)
    video_sub_list = [video_1_sub, video_2_sub, video_3_sub, video_4_sub]
    video_sub_list_len = [len(video_1_sub), len(video_2_sub), len(video_3_sub), len(video_4_sub)]
    return video_sub_list, video_sub_list_len

def preprocess_swan_df(hbn_data_path) -> pd.DataFrame:
    df_swan = pd.read_csv(os.path.join(hbn_data_path, 'SWAN_Scores.csv'))
    # delete useless last row
    df_swan = df_swan.drop(len(df_swan)-1)
    # delete all user without swan_score
    df_swan = df_swan[df_swan.SWAN_Total != '.']
    # convert string to float
    swan_scores = ['SWAN_HY', 'SWAN_IN', 'SWAN_Total']
    df_swan.loc[:, swan_scores] = df_swan.loc[:, swan_scores].astype(float)
    return df_swan

def load_swan_scores(df, hbn_data_path) -> pd.DataFrame:
    df_swan = preprocess_swan_df(hbn_data_path)
    swan_scores = ['SWAN_HY', 'SWAN_IN', 'SWAN_Total']
    df[swan_scores] = ""
    tmp_ar = np.empty((1,3))
    tmp_ar[:] = np.NaN
    for sub in df.Patient_ID.tolist():
        if df_swan[df_swan.EID == sub].empty:
            df.loc[df.Patient_ID == sub, swan_scores] = tmp_ar
        else:
            df.loc[df.Patient_ID == sub, swan_scores] = df_swan[df_swan.EID == sub][swan_scores].values

    return df

def load_label(df, hbn_data_path) -> pd.DataFrame:
    label_csv =pd.read_csv(os.path.join(hbn_data_path, 'adhd_label.csv'), sep=',')
    df['label'] = ""
    for sub in df.Patient_ID.tolist():
        if label_csv[label_csv.subId == sub]['label'].empty:
            df.loc[df.Patient_ID == sub, 'label'] = np.nan
        else:
            df.loc[df.Patient_ID == sub, 'label'] = np.unique(label_csv[label_csv.subId == sub]['label'].values)

    return df

def load_sub_information(sub_list, hbn_data_path, swan_score_ascending):
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'BasicData.csv'))
    mask = df_basic.Patient_ID.apply(lambda x: x in sub_list)
    df_basic = df_basic[mask]
    df_basic_with_score = load_swan_scores(df_basic, hbn_data_path).sort_values(by='SWAN_Total', ascending=swan_score_ascending)
    df_res = load_label(df_basic_with_score, hbn_data_path)
    df_res = load_sampling_rate(df_res, hbn_data_path)
    return df_res


def load_control_group(hbn_data_path) -> pd.DataFrame:
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'BasicData.csv'))
    df_basic = df_basic[df_basic.DX_01 == 'No Diagnosis Given']
    df_basic = df_basic[df_basic.Participant_Status == 'Complete']

    return df_basic

def load_disorder_group(disorder, hbn_data_path) -> pd.DataFrame:
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'BasicData.csv'))
    df_basic = df_basic[df_basic.Participant_Status == 'Complete']
    df_disease = df_basic[(df_basic.DX_01.str.contains(disorder)) & df_basic.DX_01.notnull() & df_basic.DX_02.isnull()]

    return df_disease
