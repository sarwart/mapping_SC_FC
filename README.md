# **Investigating structure-function coupling in the human connectome using deep learning**

We are delighted to provide the neuroscience community with a new deep learning framework to predict an individual’s brain function (functional connectivity) from their structural connectome. This repository provides the code that could be used to train a neural network using a dataset comprising of structural and functional connectivity matrices. Refer to the paper for selection of the hyper-parameters.

The rest of this document is sectioned according to files and scripts in this repository.

[**network.py**](https://github.com/sarwart/mapping_SC_FC/blob/master/network.py) provides the neural network architecture of the proposed framework (figure below)

![alt text](https://github.com/sarwart/mapping_SC_FC/blob/master/architecture.png)

[**train.py**](https://github.com/sarwart/mapping_SC_FC/blob/master/network.py) is the main script for training a neural network 

[**reload.py**](https://github.com/sarwart/mapping_SC_FC/blob/master/network.py) is a sample script for predicting functional connectivity using a pre-trained neural network 

[**example_data.mat**](https://github.com/sarwart/mapping_SC_FC/blob/master/example_data.mat) represents the sample data which consists of the 100 structural and functional connectivity matrices (the upper triangle of the connectivity matrix). This dataset is used as an input for train.py script.



