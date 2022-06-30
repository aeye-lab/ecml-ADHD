import numpy as np
import os
from scipy.ndimage import zoom
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import cv2
import seaborn as sns
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import argparse
tf.disable_v2_behavior()
sns.set_style('white')

def create_saliency_map(video_indx, model, save_path, video_path):
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_indx_to_numframe_dict = {1: 2816, 2: 5150, 3: 4265, 4: 4877}
    num_frames = video_indx_to_numframe_dict.get(video_indx)
    video_name = dict_video_indx_name_mapping.get(video_indx)
    video_dir = video_path + f'frames_{video_name}/'

    # load precomputed log density over a 1024x1024 image
    centerbias_template = np.load('centerbias.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template, (600/1024, 800/1024), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

    log_density_tmp = []
    for i in tqdm(range(num_frames)):
        img_path = video_dir  + f'frame{i}' + '.jpg'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 600))
        img = img[:,:,[2,1,0]]
        image_data = img[np.newaxis,:,:,:]

        tf.reset_default_graph()
        check_point = f'{model}.ckpt'  # DeepGaze II
        #check_point = 'ICF.ckpt'  # ICF
        new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

        input_tensor = tf.get_collection('input_tensor')[0]
        centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        log_density = tf.get_collection('log_density')[0]
        log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

        with tf.Session() as sess:
            new_saver.restore(sess, check_point)
            log_density_prediction = sess.run(log_density, {
                input_tensor: image_data,
                centerbias_tensor: centerbias_data,
            })

            #print(log_density_prediction.shape)
        log_density_tmp.append(log_density_prediction[0,:,:,:])


    log_density_prediction = np.array(log_density_tmp)# BHWC, three channels (RGB)
    print(log_density_prediction.shape)
    #save npy file
    save_values_path = save_path + model + '/' + video_name + '/'
    if not os.path.exists(save_values_path):
        os.makedirs(save_values_path)
    np.save(save_values_path + f'log_density_prediction_{video_name}.npy', log_density_prediction)



def visualization(video_indx, model, save_path, video_path):
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_indx_to_numframe_dict = {1: 2816, 2: 5150, 3: 4265, 4: 4877}
    num_frames = video_indx_to_numframe_dict.get(video_indx)
    video_name = dict_video_indx_name_mapping.get(video_indx)
    video_dir = video_path + f'frames_{video_name}'+'/'

    save_values_path = save_path + model + '/' + video_name + '/' + f'log_density_prediction_{video_name}.npy'
    log_density_prediction=np.load(save_values_path)

    #visualize model prediction
    vmax = 6e-5
    vmin = 0
    norm = Normalize(vmin = vmin, vmax = vmax)

    plots_folder = save_path + model + '/' + video_name + '/' + 'salmap/'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    # dots per inch
    dpi = 100.0
    dispsize = (800,600)
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    for i in tqdm(range(num_frames)):
        img_path = video_dir  + f'frame{i}' + '.jpg'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 600))
        img = img[:,:,[2,1,0]]
        # create a figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.gca().imshow(img, alpha=0.2)
        m = plt.gca().matshow(np.exp(log_density_prediction[i, :, :, 0]), alpha=0.6, cmap=plt.cm.RdBu)
        plt.axis('off')
        savefilename = plots_folder  + f'frame{i}' + '.jpg'
        if savefilename != None:
            fig.savefig(savefilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write saliency map file')
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
        '--save_salmap_path',
        help='save saliency map path',
        type=str,
        default='../../Data/saliency_map/'
    )

    parser.add_argument(
        '--saved_video_frame_path',
        help='saved_video_frame_path',
        type=str,
        default='../../Data/videos/'
    )

    parser.add_argument(
        '--vis_flag',
        help='Flag for visualizing the saliecny map and save in the folder',
        type=int,
        default=0
    )

    args = parser.parse_args()
    video_indx = args.video_indx
    sal_model = args.sal_model
    save_salmap_path = args.save_salmap_path
    video_path = args.saved_video_frame_path
    vis_flag = args.vis_flag


    create_saliency_map(video_indx=video_indx, model=sal_model, save_path=save_salmap_path, video_path=video_path)
    if vis_flag:
        visualization(video_indx=video_indx, model=sal_model, save_path=save_salmap_path, video_path=video_path)
