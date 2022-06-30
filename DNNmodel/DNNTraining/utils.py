import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def configure_gpu(gpu):
    #GPU = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu);
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    import tensorflow as tf;
    config = tf.compat.v1.ConfigProto(log_device_placement=True);
    config.gpu_options.per_process_gpu_memory_fraction = 0.5;
    config.gpu_options.allow_growth = True;
    tf_session = tf.compat.v1.Session(config=config)


def load_ADHD_classification_sub_info(video_name, sub_info_path):
    file_path = sub_info_path + '/sub_sel_classif.csv'
    df = pd.read_csv(file_path, sep='\t')
    df = df[df[f'video_{video_name}'] == True]
    user_list = df.Patient_ID.values.tolist()
    return df, user_list

def load_all_sub_info(video_name, sub_info_path):
    file_path = sub_info_path + '/sub_sel_all.csv'
    df = pd.read_csv(file_path, sep='\t')
    df = df[df[f'video_{video_name}'] == True]
    sub_list = df.Patient_ID.values.tolist()
    return df, sub_list

def load_X_input_files(X_files_path, video_indx, user_list, df_label, label, remove_input_channel):
    expt = {    'px_x':800,
                'px_y':600,
                'cm_x': 33.8,
                'cm_y':27.0,
                'dist':63.5,
                'sampling_rate':120}

    #video information
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_name = dict_video_indx_name_mapping.get(video_indx)

    X = []
    for sub in tqdm(user_list):
        sub_file_path = X_files_path + f'X_{sub}_DeepGazeII_Video_{video_name}.npy'
        print('Loading data from file {}'.format(sub_file_path))
        # load X file
        X_tmp = np.load(sub_file_path, allow_pickle=True)
        #remove selected input channels for ablation study
        if remove_input_channel != 'NA':
            if remove_input_channel == 'loc':
                X_tmp = X_tmp[:,[2,3]]
            elif remove_input_channel == 'dur':
                X_tmp = X_tmp[:,[0,1,3]]
            elif remove_input_channel == 'sal':
                X_tmp = X_tmp[:,[0,1,2]]

        X.append(X_tmp)

    #create labels, classification setting: binary labels 0 (healthy control), 1 (ADHD), regression setting: Swanscore
    if label == 'binary':
        Y = [df_label[df_label.Patient_ID == u].label.values for u in user_list]
    elif label == 'reg':
        Y = [df_label[df_label.Patient_ID == u].SWAN_Total.values for u in user_list]

    #exclude user with abnormal less fixations
    X_len = [i.shape[0] for i in X]
    exclude_indx = np.where(np.array(X_len)<100)[0]
    exclude_indx = sorted(exclude_indx, reverse=True)
    for indx in exclude_indx:
        X.pop(indx)
        Y.pop(indx)
        X_len.pop(indx)
        user_list.pop(indx)

    max_len = np.max(X_len)
    assert len(X)== len(Y)
    assert len(X) == len(user_list)
    Y = np.asarray(Y).astype('float32')

    return X, Y, user_list, max_len

def padding_zeros(X, pad_len):
    X = [np.pad(i, pad_width=((0, pad_len-i.shape[0]), (0, 0)), mode='constant', constant_values=0) for i in X]
    X = np.array(X)
    return X

def compute_mean_std(X, remove_input_channel):
    if remove_input_channel == 'NA':
        channel1 = [x[:,0] for x in X]
        channel1 = np.concatenate(channel1).ravel()
        channel2 = [x[:,1] for x in X]
        channel2 = np.concatenate(channel2).ravel()
        channel3 = [x[:,2] for x in X]
        channel3 = np.concatenate(channel3).ravel()
        channel4 = [x[:,3] for x in X]
        channel4 = np.concatenate(channel4).ravel()

        mean=np.array([np.mean(channel1), np.mean(channel2), np.mean(channel3), np.mean(channel4)])
        std=np.array([np.std(channel1), np.std(channel2), np.std(channel3), np.std(channel4)])

    elif remove_input_channel == 'loc':
        channel1 = [x[:,0] for x in X]
        channel1 = np.concatenate(channel1).ravel()
        channel2 = [x[:,1] for x in X]
        channel2 = np.concatenate(channel2).ravel()

        mean=np.array([np.mean(channel1), np.mean(channel2)])
        std=np.array([np.std(channel1), np.std(channel2)])


    elif remove_input_channel == 'dur' or remove_input_channel == 'sal':
        channel1 = [x[:,0] for x in X]
        channel1 = np.concatenate(channel1).ravel()
        channel2 = [x[:,1] for x in X]
        channel2 = np.concatenate(channel2).ravel()
        channel3 = [x[:,2] for x in X]
        channel3 = np.concatenate(channel3).ravel()

        mean=np.array([np.mean(channel1), np.mean(channel2), np.mean(channel3)])
        std=np.array([np.std(channel1), np.std(channel2), np.std(channel3)])

    return mean, std
