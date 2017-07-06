# SOT_track
Single object tracking, implemented from [SINT][https://arxiv.org/abs/1605.05863]

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

## How to train the SINT model
	* Refer more details to 'train/train_SINT_triplet.py'
	* example usage
	''' python
	python train_SINT_triplet.py --lmnn --rand_mirror --fine_tune --lr 0.001 --lmnn_threshd 0.9 --gpus 0
	'''