# spot-ml
New Machine learning routines for Apache Spot.
spot-ml contains routines for performing *suspicious connections* analyses on netflow data gathered from a network. These routines implement 3 different type of algorithms, One Class SVM, Isolation Forest and Location Outlier.

## Overview
As these algorithms are not support yet by the official project, it has been implemented the algorithms with the python library called scikit-learn.

The **One Class SVM** is an unsupervised novelty detection algorithm. It considers a dataset of *n* observations from the same distribution with *p* features for the training stage. Therefore, it learns a rough, close frontier delimiting the contour of the initial observations distribution, plotted in embedding *p-dimensional* space. Then, if further observations lay within the frontier-delimited subspace, they are considered as coming from the same population than the initial observations. Otherwise, if they lay outside the frontier, we can say that they are abnormal with a given confidence in our assessment. It can be configured the frontier-delimiter with the *nu* parameter, which corresponds to the probability of finding a new, but regular, observation outside the frontier.

The **Isolation Forest** is an unsupervised outlier detection algorithm. It "isolates" observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

The **Local Outlier Factor** is also an unsupervised outlier detection algorithm. It computes a score (called local outlier factor) reflecting the degree of abnormality of the observations. It measures the local density deviation of a given data point with respect to its neighbors. The idea is to detect the samples that have a substantially lower density than their neighbors. In practice the local density is obtained from the k-nearest neighbors. The LOF score of an observation is equal to the ratio of the average local density of his k-nearest neighbors, and its own local density: a normal instance is expected to have a local density similar to that of its neighbors, while abnormal data are expected to have much smaller local density.
The number k of neighbors considered, (alias parameter **n_neighbors**) is typically chosen:

  - Greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster.
  - Smaller than the maximum number of close by objects that can potentially be local outliers.

The differences between **novelty detection** and **outlier detection** algorithms lies on the training phase. While the novelty detection algorithm needs to be trained with a "clean" dataset (not polluted by outliers), the other ones are trained with data that contains outliers. Therefore, the outlier detection algorithms need to fit the central mode of the training data, ignoring the deviant observations.

# Getting Started
## Prerequisites

* Docopt
* Schema
* Pandas
* Numpy
* Scikit-learn

## Install

1. Install Python dependencies `pip install -r requirements.txt`

## Configure the /etc/spot.conf file

Additionally to the official spot.conf file, it has been added the following parameters:

- `TRAINING_PATH` : All the algorithms implements a separately training phase. So, this variable represents the output for this stage. This path is used for saving the trained network and the minimum-maximum file obtained in the normalization function. Therefore, there will be one trained network per algorithm.

    * The root path is `${HUSER}/${DSOURCE}/training`, while the path for the algorithms will be `${HUSER}/${DSOURCE}/training/algorithms`

    > The trained network name file will be `algorithm_network.plk` (e.g.: `OneClassSVM_network.plk`).


- `TESTING_PATH` : It combines `FLOW_PATH`, `DNS_PATH` and `PROXY_PATH` in a single variable. It takes the value of the allocation for the ingested data available for the Machine Learning Module.

    * The root path is `${HUSER}/${DSOURCE}/hive/y=${YR}/m=${MH}/d=${DY}/`.

- `ALGORITHM` : It represents the selected algorithm to be used by the ML module. It can take 3 values: OneClassSVM, IsolationForest or LocalOutlier (*string*). Depending on the choice, it **must** be also configured the following parameters:
    * OneClassSVM
        + `NU` : It is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. It should be in the interval (0,1] ( *float* ).
        + `KERNEL` : It specifies the kernel type to be used in the algorithm. It must be one of `linear`, `poly`, `rbf`, `sigmoid` or `precomputed` ( *string* ).
    * IsolationForest
        + `ESTIMATORS` : It is the number of base estimators in the ensemble ( *int* ).
        + `CONTAMINATION` : The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function. It must be in the interval (0, 0.5) ( *float* ).
    * LocalOutlier
        + `NEIGHBORS` : It is the number of neighbors to use by default for kneighbors queries. If it is larger than the number of samples provided, all samples will be used ( *int* ).
        + `CONTAMINATION` : The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the decision function. It must be in the interval (0, 0.5) ( *float* ).

## Prepare data for input

The schema used by the **training** and the **testing** phase is the one provided by the ingestion data via the Spot ingest. However, it is necessary to make sure the data used for the **training** stage with the **One Class SVM** algorithm is a clean dataset.


## Run a suspicious connects analysis
To run a suspicious connects analysis, execute the ml_security.sh script in the ml directory of the MLNODE.

```
./ml_security PHASE TYPE YYYYMMDD
```

Training:
```
./ml_security train flow 20171003
```

Once it is finished, there should be two files saved in the HDFS filesystem.
* Minimum-maximum values from the normalization function:
```
$HPATH/flow/training/trained_min_max.json
```
* Trained network:
```
$HPATH/flow/training/algorithms/OneClassSVM_network.plk
```

Regarding the testing phase:
```
./ml_security test flow 20180214
```
When it ends, there should be a file with the suspicious flow events from
```
$HPATH/flow/scored_results/YYYYMMDD/scores/flow_results.csv
```
It is a csv file in which network events annotated with estimated probabilities and sorted in ascending order.
