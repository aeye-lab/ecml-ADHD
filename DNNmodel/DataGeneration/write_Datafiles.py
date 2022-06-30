import os
import re
import csv
import numpy as np
from tqdm import tqdm
import json
from scipy.io import loadmat
import argparse
from joblib import Parallel, delayed
from utils import deg2pix, pix2deg, load_all_sub_info, load_px, get_event_list, create_extrac_mask


def generate_and_write_file(px_load_path, cur_save_path, pix_x, density_prediction_norm,
                           screenPX_x=800,
                           screenPX_y=600,
                           screenCM_x=33.8,
                           screenCM_y=27.0,
                           distanceCM=63.5,):

    print('Loading data from file {}'.format(px_load_path))
    #load file to a dataframe -- columns: [video_frame, Time, X_right_orig, Y_right_orig]
    X_df = load_px(px_load_path)
    #x, y pixel coordinates
    X = X_df['X_right_orig'].values
    Y = X_df['Y_right_orig'].values
    #preprocess raw eye tracking data into sequence of fixations using the Disperson-Threshold Identification algorithm
    list_dicts, event_df = get_event_list(X,Y)
    fixations = list_dicts['fixations']

    sub_X = []
    for fix in fixations:
        #x, y position
        X_tmp = X_df.iloc[fix[0] : fix[-1]+1]
        x_pix = X_tmp['X_right_orig'].values
        y_pix = X_tmp['Y_right_orig'].values
        x_mean = np.mean(x_pix)
        y_mean = np.mean(y_pix)
        x_dva = pix2deg(x_mean, screenPX_x, screenCM_x, distanceCM, adjust_origin=True)
        y_dva = pix2deg(y_mean, screenPX_y, screenCM_y, distanceCM, adjust_origin=True, reverse_axis=True)
        #fixation duration
        dur = (X_tmp.iloc[-1].Time - X_tmp.iloc[0].Time)/1000
        #extract the normalized saliency value of each fixation location
        #in case a fixation spans multiple frames, use the central frame for the saliency computation.
        frame_central = np.median(np.unique(X_tmp.video_frame.values))
        density_tmp = density_prediction_norm[int(frame_central),:,:,0]
        #create a Gaussian kernel centered at the fixation point
        extract_map = create_extrac_mask([x_mean, y_mean], x_width=pix_x, dispsize=[800, 600])
        # compute weighted saliency values
        sal_mean = np.sum(np.multiply(extract_map, density_tmp))
        sub_X.append(np.array([x_dva, y_dva, dur, sal_mean]))

    np.save(cur_save_path, np.array(sub_X))
    #print('path:', px_load_path)
    print(np.shape(np.array(sub_X)))
    print('Done with writing files')



# write npy file of model input
def write_input_files(video_indx, model, salmap_path, px_path, save_dir, sub_info_path):
    #video information
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_indx_to_numframe_dict = {1: 2816, 2: 5150, 3: 4265, 4: 4877}
    video_name = dict_video_indx_name_mapping.get(video_indx)
    num_frames = video_indx_to_numframe_dict.get(video_indx)
    #Experiment configuration
    expt = {    'px_x':800, #screen width in pixels
                'px_y':600, #screen height in pixels
                'cm_x': 33.8, #screen width in cm
                'cm_y':27.0, #screen height in cm
                'dist':63.5, #eye-to-screen distance in cm
                'sampling_rate':120} #sampling rate in Hz

    #define parameter of gaussian blob to account for the parafoveal information
    deg = 9 # parafovea: 4.5 deg on each side around the fixation point
    pix_x = int(deg2pix(deg, expt['px_x'],expt['cm_x'],expt['dist']))
    pix_y = int(deg2pix(deg, expt['px_y'],expt['cm_y'],expt['dist']))
    #load saliency map values
    saved_saliency_map_path = salmap_path + model + '/' + video_name + '/' + f'log_density_prediction_{video_name}.npy'
    log_density_prediction=np.load(saved_saliency_map_path)
    density_prediction = np.exp(log_density_prediction) #(num_frames, image_height, image_width, 1)
    #normalize saliency values for each image to 0-1
    density_prediction_tmp = density_prediction.reshape((density_prediction.shape[0], -1))
    density_prediction_tmp -= np.min(density_prediction_tmp, axis = 1, keepdims=True)
    density_prediction_tmp /= np.max(density_prediction_tmp, axis = 1, keepdims=True)
    density_prediction_norm = density_prediction_tmp.reshape(density_prediction.shape[0], density_prediction.shape[1], density_prediction.shape[2],1)

    #load subjects information
    df_sub, sub_list = load_all_sub_info(video_name, sub_info_path)

    #generate files in parallel
    load_files = []
    X_save_files = []
    for i in tqdm(np.arange(len(sub_list))):
        sub = sub_list[i]
        cur_save_path =  save_dir + f'X_{sub}_{model}_Video_{video_name}.npy'
        if os.path.exists(cur_save_path):
            continue
        px_load_path = px_path  + f'X_px_{sub}_Video_{video_name}.npy'
        print('Loading data from file {}'.format(px_load_path))
        load_files.append(px_load_path)
        X_save_files.append(cur_save_path)

    n_files = 10
    for i in tqdm(range(0, np.int(np.ceil(len(load_files)/n_files)))):
        start_idx = i * n_files
        if start_idx + n_files <= len(load_files):
            end_idx = start_idx + n_files
        else:
            end_idx = len(load_files)
        files_done = Parallel(n_jobs=n_files)(delayed(generate_and_write_file)(load_files[i],
                               X_save_files[i], pix_x, density_prediction_norm) for i in range(start_idx, end_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write Data Files')
    parser.add_argument(
        '--video_indx',
        help='video index to name: 1: Diary_of_a_Wimpy_Kid_Trailer, 2: Fractals, 3: Despicable_Me, 4:The_Present',
        type=int,
        default=2
    )

    parser.add_argument(
        '--sal_model',
        help='saliency model name',
        type=str,
        default='DeepGazeII'
    )

    parser.add_argument(
        '--saved_salmap_path',
        help='saved saliency map path',
        type=str,
        default='../../Data/saliency_map/'
    )

    parser.add_argument(
        '--pix_files_dir',
        help='Directory for the preprocessed pixel files (after downsampling and missing value processing)',
        type=str,
        default='../../Data/X_px/'
    )

    parser.add_argument(
        '--save_dir',
        help='Directory for saving new input files',
        type=str,
        default='../../Data/X_input/'
    )

    parser.add_argument(
        '--sub_info_path',
        help='Directory for saving subject information',
        type=str,
        default='../../Data/sub_info/'
    )

    args = parser.parse_args()
    video_indx = args.video_indx
    sal_model = args.sal_model
    salmap_path = args.saved_salmap_path
    px_path = args.pix_files_dir
    save_dir = args.save_dir
    sub_info_path = args.sub_info_path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #video index 2 name: 1: Diary_of_a_Wimpy_Kid_Trailer, 2: Fractals, 3: Despicable_Me, 4:The_Present
    write_input_files(video_indx=video_indx,
                        model=sal_model,
                        salmap_path=salmap_path,
                        px_path=px_path,
                        save_dir=save_dir,
                        sub_info_path=sub_info_path)
