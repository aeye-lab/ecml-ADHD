import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.backend import clear_session
from model import CNN_reg, CNN_classif
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import argparse
import json
from utils import configure_gpu, load_ADHD_classification_sub_info, load_all_sub_info, load_X_input_files, padding_zeros, compute_mean_std



def model_pre_train(test_user, video_indx, sal_model, remove_input_channel, pad_len, sub_info_path, input_dir, gpu):
    print(f'start evaluate with video{video_indx}, saliency mode: {sal_model}')
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_name = dict_video_indx_name_mapping.get(video_indx)
    #load subjects information
    df_sub, sub_list = load_all_sub_info(video_name, sub_info_path)
    #exclude person that do not have Swan Score measure
    df_sub = df_sub[df_sub['SWAN_Total'].notna()]
    #exclude person in test data
    df_sub = df_sub[~df_sub.Patient_ID.isin(test_user)]
    user_list = df_sub.Patient_ID.values.tolist()
    #print('num of person in pre traing:', len(user_list))

    #load input files for subjects and the corresponding labels
    #user_list is updated, users with abnormal less fixations are excluded
    #Returns the maximum value of the number of fixation
    X, Y, user_list, max_len = load_X_input_files(input_dir,
                                                    video_indx,
                                                    user_list,
                                                    df_sub,
                                                    label='reg',
                                                    remove_input_channel=remove_input_channel)

    # make training and test data include similar range of Swan score,
    # avoid e.g. training data only include person with Swan score larger than 0
    sign = Y.copy()
    sign[sign>=0.5] = 1
    sign[sign<0.5] = 0

    n_folds=10
    # create train validation split for training the models
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, val_idx in tqdm(skf.split(X, sign)):
        break

    X_train_net = [X[idx] for idx in train_idx]
    X_val_net = [X[idx] for idx in val_idx]
    Y_train_net, Y_val_net = Y[train_idx,:], Y[val_idx,:]

    #calculate mean and std for z-score normalization
    mean, std = compute_mean_std(X_train_net, remove_input_channel)

    #apply normalization
    X_train_net = [(x - mean)/std for x in X_train_net]
    X_val_net = [(x - mean)/std for x in X_val_net]

    #padding zeros
    if pad_len>max_len:
        max_len = pad_len

    X_train_net = padding_zeros(X_train_net,  pad_len=max_len)
    X_val_net = padding_zeros(X_val_net,  pad_len=max_len)

    # clear tensorflow session
    clear_session()
    print('Pre-train the model.')
    DNN_model = CNN_reg(seq_len=X_train_net.shape[1],channels=X_train_net.shape[2])
    model = DNN_model.train(X_train_net, Y_train_net, X_val_net, Y_val_net)

    return model, max_len, mean, std







def evaluate(video_indx, sal_model, remove_input_channel, pretrain, sub_info_path, input_dir, n_rounds, n_folds, results_save_dir, save_weights_flag, gpu):
    print(f'start evaluate with video{video_indx}, saliency mode: {sal_model}')

    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_name = dict_video_indx_name_mapping.get(video_indx)

    #load subject information for ADHD classification setting
    df_sub_classif, user_list = load_ADHD_classification_sub_info(video_name, sub_info_path)
    #load input files for subjects and the corresponding labels
    #user_list is updated, users with abnormal less fixations are excluded
    #Returns the maximum value of the number of fixation
    X, Y, user_list, max_len = load_X_input_files(input_dir,
                                                    video_indx,
                                                    user_list,
                                                    df_sub_classif,
                                                    label='binary',
                                                    remove_input_channel=remove_input_channel)
    print('number of ADHD vesus control people:', np.count_nonzero(Y), Y.shape[0]-np.count_nonzero(Y))

    auc_rounds = []
    fold_counter = 1
    for i in range(n_rounds):
        aucs=[]
        #split users to n folds for cross validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        fold_idx=1
        for train_idx, test_idx in tqdm(skf.split(X, Y)):
            print('Starting evaluation fold {}/{}...'.format(fold_idx, n_folds))
            X_train = [X[idx] for idx in train_idx]
            X_test = [X[idx] for idx in test_idx]
            Y_train, Y_test = Y[train_idx,:], Y[test_idx,:]
            test_user = [user_list[i] for i in test_idx]
            train_user = [user_list[i] for i in train_idx]

            # create train validation split for training the models:
            skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
            for train_index, val_index in skf_val.split(X_train, Y_train):
                # we only evaluate a single fold
                break

            X_train_net = [X_train[idx] for idx in train_index]
            X_val_net = [X_train[idx] for idx in val_index]
            Y_train_net, Y_val_net = Y_train[train_index,:], Y_train[val_index,:]
            val_user = [train_user[i] for i in val_index]

            # clear tensorflow session
            clear_session()
            configure_gpu(gpu)
            print('Build and train model.')
            if pretrain==True:
                #pretrain the model
                #test and validation data are excluded for pretraining to ensure that that they are not seen during training
                exclude_user = test_user + val_user
                pretrain_model, max_len, mean, std = model_pre_train(test_user = exclude_user,
                                                                                video_indx=video_indx,
                                                                                sal_model = sal_model,
                                                                                remove_input_channel = remove_input_channel,
                                                                                pad_len=max_len,
                                                                                sub_info_path=sub_info_path,
                                                                                input_dir=input_dir,
                                                                                gpu=gpu)


            else:
                pretrain_model=None
                #calculate mean and std for z-score normalization
                mean, std = compute_mean_std(X_train_net, remove_input_channel)

            #apply normalization
            X_train_net = [(x - mean)/std for x in X_train_net]
            X_val_net = [(x - mean)/std for x in X_val_net]
            X_test = [(x - mean)/std for x in X_test]

            X_train_net = padding_zeros(X_train_net,  pad_len=max_len)
            X_val_net = padding_zeros(X_val_net,  pad_len=max_len)
            X_test = padding_zeros(X_test,  pad_len=max_len)

            DNN_model = CNN_classif(seq_len=X_train_net.shape[1],channels=X_train_net.shape[2])
            hist = DNN_model.train(X_train_net, Y_train_net,
                                    X_val_net, Y_val_net,
                                    pretrain_model = pretrain_model,
                                    save_weights_flag = save_weights_flag,
                                    fold_counter = fold_counter)

            pred_test = np.squeeze(DNN_model.model.predict([X_test]))
            fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred_test)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc.tolist())
            print(aucs)
            fold_idx +=1
            fold_counter +=1
        auc_rounds.append(aucs)

    print('Mean AUC:', np.mean(auc_rounds))
    print('Standanrd error:', np.std(auc_rounds)/(n_rounds*n_folds))
    #save file
    dic={'auc': auc_rounds}
    cur_save_path = results_save_dir + f'res_Video{video_indx}_pretrain{pretrain}_remove{remove_input_channel}_{num_folds}folds_{num_iter}iter.json'
    isExist = os.path.exists(cur_save_path)
    if not isExist:
        os.makedirs(cur_save_path)
    with open(cur_save_path, 'w') as fp:
        json.dump(dic, fp, indent = 4)

    print('Finish evaluation, results saved.')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hyperparameter tuning')
    parser.add_argument(
        '--video_indx',
        help='video_index',
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
        '--remove_input_channel',
        help='When using all input channels set to NA. For the ablation study, one of the four input channels is removed, the valid input channels to be removed are loc, dur, sal.',
        type=str,
        default='NA'
    )

    parser.add_argument(
        '--pre_train',
        help='if do pretraining',
        type=int,
        default=True
    )

    parser.add_argument(
        '--sub_info_path',
        help='Directory for saving subject information',
        type=str,
        default='./Data/sub_info/'
    )

    parser.add_argument(
        '--input_dir',
        help='Directory for saved input files',
        type=str,
        default='./Data/X_input/'
    )

    parser.add_argument(
        '--num_iter',
        help='Number of iterations to run',
        type=int,
        default=10
    )

    parser.add_argument(
        '--num_folds',
        help='Number of splitting folds for cross-validation',
        type=int,
        default=10
    )

    parser.add_argument(
        '--results_save_dir',
        help='directory for saving results',
        type=str,
        default='./results/'
    )

    parser.add_argument(
        '--save_weights_flag',
        help='Flag to save weights for all the models',
        type=int,
        default=0
    )


    parser.add_argument(
        '--gpu',
        help='gpu_index',
        type=int,
        default=6
    )

    args = parser.parse_args()
    video_indx = args.video_indx
    sal_model = args.sal_model
    remove_input_channel = args.remove_input_channel
    assert remove_input_channel in ["NA", "loc", "dur", "sal"], "No such input channels to be removed"
    pre_train = args.pre_train
    sub_info_path = args.sub_info_path
    input_dir = args.input_dir
    num_iter = args.num_iter
    num_folds = args.num_folds
    results_save_dir = args.results_save_dir
    save_weights_flag = args.save_weights_flag
    gpu_indx = args.gpu


    evaluate(video_indx=video_indx,
                sal_model=sal_model,
                remove_input_channel=remove_input_channel,
                pretrain=pre_train,
                sub_info_path=sub_info_path,
                input_dir=input_dir,
                n_rounds=num_iter,
                n_folds=num_folds,
                results_save_dir=results_save_dir,
                save_weights_flag=save_weights_flag,
                gpu=gpu_indx)
