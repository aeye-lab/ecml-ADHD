# Detection of ADHD based on Eye Movements during Natural Viewing
[![paper](https://img.shields.io/static/v1?label=paper&message=download%20link&color=brightgreen)](https://arxiv.org/abs/2207.01377)

In this paper, we explore whether Attention-deficit/hyperactivity disorder (ADHD) can be detected based on recorded eye movements together with information about the video stimulus in a free-viewing task

## Setup

Clone repository:

```
git clone git@github.com:aeye-lab/ecml-ADHD
```

or

```
git clone https://github.com/aeye-lab/ecml-ADHD
```

and change to the cloned repo via `cd ecml-ADHD`.


Install dependencies:

```
pip install -r requirements.txt
```

## Run Experiments

### Prepare saliency maps
Please download videos from [here](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/EEG_Eyetracking_Protocol.html), and break it down into frames (e.g. use ffmpeg packages) and put them under the folder /Data/videos/frames_{video_name}/

We use a state-of-the-art saliency model, DeepGazeII to compute salinecy maps for our video stimuli. Download the files you need to run the DeepGazeII model https://drive.google.com/file/d/1kYUwoatqQUS5EabeeSDc6gRmCysnVZ6N/view, and stored under the folder /DNN_model/DataGeneration/

To generate the saliency maps for all videos, run
```
bash gen_saliency_map_data.sh
```

### Prepare model input files

To generate model input files, run

```
bash gen_model_input_data.sh
```

### Run models

```
bash run_models.sh
```


## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@inproceedings{deng2023detection,
  title={Detection of ADHD based on eye movements during natural viewing},
  author={Deng, Shuwen and Prasse, Paul and Reich, David R and Dziemian, Sabine and Stegenwallner-Sch{\"u}tz, Maja and Krakowczyk, Daniel and Makowski, Silvia and Langer, Nicolas and Scheffer, Tobias and J{\"a}ger, Lena A},
  booktitle={Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022, Grenoble, France, September 19--23, 2022, Proceedings, Part VI},
  pages={403--418},
  year={2023},
  organization={Springer}
}
```
