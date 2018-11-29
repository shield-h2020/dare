# Threat Classifier DL
#
#
[![SHIELD](https://www.shield-h2020.eu/shield-h2020/img/logo/shield_navbar.png)](https://www.shield-h2020.eu/shield-h2020/img/logo/shield_navbar.png)

## Description

__Threat Classifier DL is an open-source, netflow traffic classifier based on Artificial Neural Networks. It utilizes an implementation of the MultiLayer Perceptron to assign threat labels to netflow activity.__

MLP is a class of feedforward artificial neural networks, consisting of at least three layers of nodes (input, hidden, and output layers). In MLP, each neuron unit calculates the linear combination of its real-valued inputs and passes it through a threshold activation function. 

The proposed architecture involves an input layer, a batch-normalisation layer, two hidden dense layers consisting of 36 and 12 nodes respectively and the output layer. The rectified linear unit (ReLU) is chosen as the activation function of the hidden dense layers, while Softmax is used for the output layer. The model was trained for 10 epochs using the Adagrad optimizer and categorical cross-entropy as loss function. The selected MLP model was integrated in the Cognitive DA module of the DARE

Threat Classifier DL is part of SHIELD's Cognitive DA module and is capable of assigning threat labels to detected anomalies. It serves as an alternative to the Random Forest classifier and further expands our data analytics solutions. It is developed using the desktop version of the Deep Learning Studio, which is compatible with a number of open-source programming frameworks, popularly used in artificial neural networks, including MXNet and Google's TensorFlow. 

## Framework

Threat Classifier DL uses a number of open-source frameworks and libraries to work properly:

* [Tensorflow] - An open-source software library for dataflow programming across a range of tasks.
* [Keras] - An open source neural network library written in Python, capable of running on top of TensorFlow
* [Deep Learning Studio] - A software tool that aims to simplify the creation of deep learning models used in artificial intelligence.


## Installation

The installation procedure assumes an established CDH Cluster (Cloudera Express >= 5.12) and Python 3 (>=3.6).
#
The repository contains the following:
- `config.yaml`, that describes the architecture of the Keras model
- `model.h5`, which contains the weights of the model stored in an HDF5/H5 file, a popular format to store structured data.
- `mapping.pkl`, a pickled dictionary for mapping characters to integers
- `test.py`, the python script used to use the trained model for classification of netflow data. 
- `requirements.txt`, a file that can be used to pip install all necessary Python3 libraries.
- `LICENCE.md`, The Apache2 Licence.
- `README.md`, this file, containing more detailed instructions about the module.

#### Installation Steps:
*  Clone or download this repository to a node of your CDH cluster. Move inside this folder.
*  Run these commands to install the necessary packages: 
    `
    $ sudo apt-get install python3 python3-pip python3-scipy python3-dev python3-h5py python3-pillow python3-pandas
    `
    `sh
     $ sudo apt-get install libblas-dev liblapack-dev
    ` 
*  Install the necessary Python requirements (You can optionally use virtualenv to avoid interfering with your current Python version.)
    `sh
     $ pip install -r requirements.txt
    `


## Usage
The classifier is able to assign threat labels to netflow data, using the following script:

`
$ python3 test.py <path/to/local/netflow.csv> tensorflow
`
Upon completion, a file named `test_results.csv` will be created on the same folder, containing the classification results. 
 
### Todos
- Connect classification procedure with anomaly detection pipeline 

License
----
Apache2


[![Funded by EU](https://www.shield-h2020.eu/shield-h2020/img/EC-H2020.png)](https://www.shield-h2020.eu/shield-h2020/img/EC-H2020.png)
