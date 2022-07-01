# File to run models
#video index to name: 1: Diary_of_a_Wimpy_Kid_Trailer, 2: Fractals, 3: Despicable_Me, 4:The_Present
#python ./DNNmodel/DNNTraining/main.py --video_indx=1 --sal_model="DeepGazeII" --remove_input_channel="NA" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
python ./DNNmodel/DNNTraining/main.py --video_indx=2 --sal_model="DeepGazeII" --remove_input_channel="NA" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=6
#python ./DNNmodel/DNNTraining/main.py --video_indx=3 --sal_model="DeepGazeII" --remove_input_channel="NA" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=4 --sal_model="DeepGazeII" --remove_input_channel="NA" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
#
#
#set pre_train flag to 0 if no training is needed.
#python ./DNNmodel/DNNTraining/main.py --video_indx=1 --sal_model="DeepGazeII" --pre_train=0 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=2 --sal_model="DeepGazeII" --pre_train=0 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=3 --sal_model="DeepGazeII" --pre_train=0 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=4 --sal_model="DeepGazeII" --pre_train=0 --num_iter=10 --num_folds=10 --gpu=0
#
#
#ablation study set argument "remove_input_channel" to ["loc", "dur", "sal"] to remove (x,y), (fix duration), (saliency value) individually
#python ./DNNmodel/DNNTraining/main.py --video_indx=2 --sal_model="DeepGazeII" --remove_input_channel="loc" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=2 --sal_model="DeepGazeII" --remove_input_channel="dur" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
#python ./DNNmodel/DNNTraining/main.py --video_indx=2 --sal_model="DeepGazeII" --remove_input_channel="sal" --pre_train=1 --num_iter=10 --num_folds=10 --gpu=0
