import numpy as np
import pandas as pd
from utils import load_HBN_sub_info, load_sampling_rate, count_subject_for_video_task, load_sub_information, load_control_group, load_disorder_group
import argparse
import os


def write_sub_sel_SWAN_files(hbn_data_path, save_data_path):
    df_sub = load_HBN_sub_info(hbn_data_path)#2216

    #exclusion criterion 1: video resolution must be 800 * 600
    df_tmp = pd.read_csv(os.path.join(hbn_data_path, 'VideoResolution.csv'))
    sub_list = set(df_tmp[df_tmp.Resolution == 800600].IDs)
    mask = df_sub.Patient_ID.apply(lambda x: x in sub_list)
    df_sub = df_sub[mask]

    #exclusion criterion 2: the sampling rate must be 60 or 120 Hz
    df_sub = load_sampling_rate(df_sub, hbn_data_path)
    delete_row = df_sub[(df_sub.sampling_rate != 60) & (df_sub.sampling_rate != 120)].index
    df_sub = df_sub.drop(delete_row)

    #exclusion criterion 3: the data must be recorded before Summer 2019
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'PhenoDataFinal.csv'))
    df_basic = df_basic[df_basic.Enroll_Year<='2019']
    df_basic = df_basic[~((df_basic.Enroll_Year=='2019') & (df_basic.Enroll_Season!='Spring'))]
    user_basic = df_basic.ID.values.tolist()
    df_sub = df_sub[df_sub.Patient_ID.isin(user_basic)]

    #exclusion criterion 4: participant age should be larger than 6
    df_sub = df_sub[df_sub['Age']>=(int(6))]

    user_all = np.unique(df_sub.Patient_ID).tolist()

    #select subjects for each single video,
    #for some participants, recordings are available only from a subset of the four videos
    #exclusion criterion 5: the tracker loss must be less then 10%
    #exclusion criterion 6: data duration equals video duration
    video_sub_list, video_sub_list_len = count_subject_for_video_task(user_all, threshold_tl=0.1, hbn_data_path=hbn_data_path)
    print(video_sub_list_len)
    #threshold 0.1: [1076, 345, 971, 603]
    #1246 in total

    #make a summarize file that include all subjects with their corresponding valid video recordings
    users = []
    for i in range(4):
        users = users + video_sub_list[i]
    users = np.unique(users)
    print(len(users))
    df_res = load_sub_information(users, hbn_data_path, swan_score_ascending=True)

    #add new columns to indicate the data quality of each video for every subject,
    #video with 'True' will be used in our model training and testing
    #user list order changed, recompute the user list
    users = df_res.Patient_ID.values.tolist()
    df_res['video_Diary_of_a_Wimpy_Kid_Trailer'] = [True if u in video_sub_list[0] else False for u in users]
    df_res['video_Fractals'] = [True if u in video_sub_list[1] else False for u in users]
    df_res['video_Despicable_Me'] = [True if u in video_sub_list[2] else False for u in users]
    df_res['video_The_Present'] = [True if u in video_sub_list[3] else False for u in users]
    df_res.to_csv(save_data_path + f'sub_sel_all.csv', sep='\t', header=True, index=False)
    print('Finish generating SWAN prediction dataset subject list.')


def write_sub_sel_classif_files(hbn_data_path, save_data_path):
    #load healthy people
    df_control = load_control_group(hbn_data_path)
    #load adhd people
    df_ADHD = load_disorder_group('ADHD', hbn_data_path)

    #exclusion criterion 1: video resolution must be 800 * 600
    df_tmp = pd.read_csv(os.path.join(hbn_data_path, 'VideoResolution.csv'))
    sub_list = set(df_tmp[df_tmp.Resolution == 800600].IDs)
    mask_control = df_control.Patient_ID.apply(lambda x: x in sub_list)
    df_control = df_control[mask_control]
    mask_adhd = df_ADHD.Patient_ID.apply(lambda x: x in sub_list)
    df_ADHD = df_ADHD[mask_adhd]

    #exclusion criterion 2: the sampling rate must be 60 or 120 Hz
    df_control = load_sampling_rate(df_control, hbn_data_path)
    delete_row = df_control[(df_control.sampling_rate != 60) & (df_control.sampling_rate != 120)].index
    df_control = df_control.drop(delete_row)

    df_ADHD = load_sampling_rate(df_ADHD, hbn_data_path)
    delete_row = df_ADHD[(df_ADHD.sampling_rate != 60) & (df_ADHD.sampling_rate != 120)].index
    df_ADHD = df_ADHD.drop(delete_row)

    #exclusion criterion 3: the data must be recorded before Summer 2019
    df_basic = pd.read_csv(os.path.join(hbn_data_path, 'PhenoDataFinal.csv'))
    df_basic = df_basic[df_basic.Enroll_Year<='2019']
    df_basic = df_basic[~((df_basic.Enroll_Year=='2019') & (df_basic.Enroll_Season!='Spring'))]
    user_basic = df_basic.ID.values.tolist()
    df_control = df_control[df_control.Patient_ID.isin(user_basic)]
    df_ADHD = df_ADHD[df_ADHD.Patient_ID.isin(user_basic)]

    #exclusion criterion 4: participant age should be larger than 6
    df_control = df_control[ df_control['Age']>=(int(6))]
    df_ADHD = df_ADHD[df_ADHD['Age']>=(int(6))]

    user_control = np.unique(df_control.Patient_ID).tolist()
    user_ADHD = np.unique(df_ADHD.Patient_ID).tolist()

    #select subjects for each single video,
    #for some participants, recordings are available only from a subset of the four videos
    #exclusion criterion 5: the tracker loss must be less then 10%
    #exclusion criterion 6: data duration equals video duration
    video_sub_list_control, video_sub_list_len_control = count_subject_for_video_task(user_control, threshold_tl=0.1, hbn_data_path=hbn_data_path)
    print(video_sub_list_len_control)
    video_sub_list_ADHD, video_sub_list_len_ADHD = count_subject_for_video_task(user_ADHD, threshold_tl=0.1, hbn_data_path=hbn_data_path)
    print(video_sub_list_len_ADHD)
    #threshold 0.1: normal[138, 19, 129, 48], ADHD[203, 48, 187, 111]

    #make a summarize file that include all subjects with their corresponding valid video recordings
    users = []
    for i in range(4):
        users = users + video_sub_list_control[i] + video_sub_list_ADHD[i]

    users = np.unique(users)
    df_res = load_sub_information(users, hbn_data_path, swan_score_ascending=True)

    #add new columns to indicate the data quality of each video for every subject,
    #video with 'True' will be used in our model training and testing
    #user list order changed, recompute the user list
    users = df_res.Patient_ID.values.tolist()
    df_res['video_Diary_of_a_Wimpy_Kid_Trailer'] = [True if u in video_sub_list_control[0] or u in video_sub_list_ADHD[0] else False for u in users]
    df_res['video_Fractals'] = [True if u in video_sub_list_control[1] or u in video_sub_list_ADHD[1] else False for u in users]
    df_res['video_Despicable_Me'] = [True if u in video_sub_list_control[2] or u in video_sub_list_ADHD[2] else False for u in users]
    df_res['video_The_Present'] = [True if u in video_sub_list_control[3] or u in video_sub_list_ADHD[3] else False for u in users]
    df_res.to_csv(save_data_path + f'sub_sel_classif.csv', sep='\t', header=True, index=False)
    print('Finish generating ADHD classification dataset subject list.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write subject selection files')
    parser.add_argument(
        '--setting',
        help='prepare SWAN prediction dataset subject list (SWAN) or ADHD classification dataset subject list (classif) or both (all)',
        type=str,
        default='all'
    )

    parser.add_argument(
        '--hbn_data_path',
        help='Directory for saving HBN data',
        type=str,
        default='/mnt/projekte/pmlcluster/aeye/HBN/'
    )

    parser.add_argument(
        '--save_dir',
        help='Directory for saving new files',
        type=str,
        default='../../Data/'
    )

    args = parser.parse_args()
    setting = args.setting
    hbn_data_path = args.hbn_data_path
    save_dir = args.save_dir

    if setting == 'SWAN':
        write_sub_sel_SWAN_files(hbn_data_path, save_dir)
    elif setting == 'classif':
        write_sub_sel_classif_files(hbn_data_path, save_dir)
    elif setting == 'all':
        write_sub_sel_SWAN_files(hbn_data_path, save_dir)
        write_sub_sel_classif_files(hbn_data_path, save_dir)
