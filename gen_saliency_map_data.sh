# File to generate saliency maps for videos
#video index to name: 1: Diary_of_a_Wimpy_Kid_Trailer, 2: Fractals, 3: Despicable_Me, 4:The_Present
#if vis_flag=1: generate visualized saliency maps and save
python ./DNNmodel/DataGeneration/create_saliency_map.py --video_indx=1 --sal_model="DeepGazeII" --save_salmap_path="./Data/saliency_map/" --saved_video_frame_path="./Data/videos/" --vis_flag=0
python ./DNNmodel/DataGeneration/create_saliency_map.py --video_indx=2 --sal_model="DeepGazeII" --save_salmap_path="./Data/saliency_map/" --saved_video_frame_path="./Data/videos/" --vis_flag=0
python ./DNNmodel/DataGeneration/create_saliency_map.py --video_indx=3 --sal_model="DeepGazeII" --save_salmap_path="./Data/saliency_map/" --saved_video_frame_path="./Data/videos/" --vis_flag=0
python ./DNNmodel/DataGeneration/create_saliency_map.py --video_indx=4 --sal_model="DeepGazeII" --save_salmap_path="./Data/saliency_map/" --saved_video_frame_path="./Data/videos/" --vis_flag=0
