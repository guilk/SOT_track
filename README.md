# SOT_track
Single object tracking, implemented from [SINT](https://arxiv.org/abs/1605.05863)

## Requirements
* mxnet: 0.9.5
* numpy
* scipy

## Model
* Beijing Nas: '/data01/SOT/models'
* This folder includes all the models.

## Datasets
* ALOV++ dataset is used for training, the seqs and annotations are located at '/data01/SOT/datasets/ALOV'
* OTB dataset is used for evaluation, the seqs and annotations are located at '/data01/SOT/datasets/OTB'

## Train the model
* Refer more details to 'train/train_SINT_triplet.py'
* example usage
```shell
python train_SINT_triplet.py --lmnn --rand_mirror --fine_tune --lr 0.001 --lmnn_threshd 0.9 --gpus 0
```

## Evaluate the model
* The 'test/' folder is used to test the trained model on OTB dataset. The tracking results will be outputted to a directory.
* example usage
```shell
python eval_SINT_train.py --overlapthresh 0.6 --topK 5 --numangles 20
```
* The 'evaluation/' folder is used to compare tracking results with other trackers. More details can be found from [Visual Tracker Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/)
* example usage
```matlab
src_root = '/dir/to/your/generated/tracking_results'
prepare_for_evaluation(src_root)
```
Then add your tracker to '/evaluation/util/configTrackers.m'
```matlab
main_running.m % generate figures in '/evaluation/figs/overall' folder
main_seperate_plot.m % used for debug, will generate figures in '/evaluation/figs/overall_seperate' folder
```

