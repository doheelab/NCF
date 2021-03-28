# Neural Collaborative Filtering (NCF)

## A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17

This repo provides a PyTorch implementation and experiment codes for NCF.

## Dataset

Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data)

## Reference

Yangyang Guo's [repo](https://github.com/guoyang9/NCF)

## The requirements are as follows:
	* python==3.6
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==1.0.1
	* gensim==3.7.1
	* tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Example to run:
```
python MLP.py
```


<!-- Models | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ | --------------- | ----------------- | --------------- | -----------------
MLP    | 0.692 | 0.425 | 0.868 | 0.542
GMF    | - | - | - | -
NeuMF (without pre-training) | 0.701 | 0.425 | 0.870 | 0.549
NeuMF (with pre-training)	 | 0.726 | 0.445 | 0.879 | 0.555


This pytorch code:

Models | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ | --------------- | ----------------- | --------------- | -----------------
MLP    | 0.691 | 0.416 | 0.866 | 0.537
GMF    | 0.708 | 0.429 | 0.867 | 0.546
NeuMF (without pre-training) | 0.701 | 0.424 | 0.867 | 0.544
NeuMF (with pre-training)	 | 0.720 | 0.439 | 0.879 | 0.555 -->
