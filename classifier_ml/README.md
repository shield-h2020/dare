# Threat Classifier ML
#
#
[![SHIELD](https://www.shield-h2020.eu/shield-h2020/img/logo/shield_navbar.png)](https://www.shield-h2020.eu/shield-h2020/img/logo/shield_navbar.png)

## Description

__Threat Classifier ML is an open-source, scalable, netflow traffic classifier for Big Data environments, based on the Random forest machine-learning algorithm.__

Random Forest is an ensemble supervised method used for classification, that constructs a multitude of decision trees at training time and outputs the mode of the classes of the individual trees as the final class.

Threat Classifier ML is part of SHIELD's Cognitive DA module and is capable of assigning threat labels to detected anomalies. To this end, it is optimized to work both in collaboration with the available anomaly detection implementations of the Cognitive DA module (e.g. Autoencoder, Apache Spot etc.), as well as independently as a standalone component.

## Framework

Threat Classifier ML uses a number of open-source frameworks and libraries to work properly:

* [CDH] - Cloudera Distribution of Hadoop
* [Apache Spark] - A unified analytics engine for big data processing.
* [MLLib] - Apache Spark's scalable machine learning library, with APIs in Java, Scala, Python, and R.


## Installation

The installation procedure assumes an established CDH Cluster (Cloudera Express >= 5.12) with Spark 2 (>= 2.1) and Python 3 (>=3.6).
#
The repository contains the following:
- A folder named `trainedRF_model_spotformat`, that contains the stages of the Spark pipeline of the pretrained model.
- `threat_classifier0.4.py`, the source code of the classifier in PySpark.
- `rmq_sender.py`, a Python script to publish the classification results to the Recommendation Engine and the Security Dashboard.
- `ml.conf`, that contains the configuration of the Spark shell required to run the classifier
- `classifier3.sh`, a bash script to automate the classifier execution (train/test) procedure
- `publish.sh`, a bash script to automate the message publishing.
-  `sim_anom_class_pub.sh` and  `b_sim_anom_class_pub.sh`, scripts that automate the whole analysis procedure. The first one only classifies the detected anomalies from the anomaly detection phase (e.g. Autoencoder), while the second one can read directly and classify all the netflow logs from HiveDB. In both cases, the results are published via the RabbitMQ queue to Dashboard and Remediation Engine. 
- `requirements.txt`, a file that can be used to pip install all necessary Python3 libraries.
- `LICENCE.md`, The Apache2 Licence.
- `README.md`, this file, containing more detailed instructions about the module.

#### Installation Steps:
1. Clone or download this repository to a node of your CDH cluster. Move inside this folder.
2. Install the necessary Python requirements (You can optionally use virtualenv to avoid interfering with your current Python version.)
    ```sh
     $ pip install -r requirements.txt
    ```
3. Transfer the `trainedRF_model_spotformat` folder to path in HDFS. This path will be configured to store the results of the classification phase, e.g.:
    ```sh
     $ hdfs dfs -copyFromLocal /path/in/local/threat_classifier_ml /hdfs/user/spotuser/rf
    ```
4. Configure the paths of the `threat_classifier0.4.py`, with the updated HDFS paths (see lines: 158, 255)
5. Make __all__ bash scripts (*.sh) of this repository executable, by modifying the file permissions. For example:
    ```sh
     $ chmod +x classifier3.sh
    ```
6. Each one of the bash scripts (`classifier3.sh`, `publish.sh`, `sim_anom_class_pub.sh` and `b_sim_anom_class_pub.sh`) contains absolute paths to the different components that should be configured manually prior to running the classifier. Following the comments inside each bash script, please make sure that these paths exist and are reachable. 

## Usage
The classifier includes automations for running either a complete detection pipeline:
|ingested data > anomaly detection > *classification* > publishing of results|,  
or independent standalone processes (e.g: train a new model). 

Below are listed all different functionalities:

- To __train__ a Random Forest of trees and create a new classification model using labeled data: 
     ```sh
     $ ./classifier3.sh train /path/to/HDFS/labeled_netflow.csv  <#trees (optional)>
    ```
    e.g:
    ```sh
     $ ./classifier3.sh train /user/spotuser/rf/netflow.csv  50
    ```
- To __predict__ the unknown labels of netflow data (threat classification):
    ```sh
     $ ./classifier3.sh predict /path/to/HDFS/flows.csv
    ```
- To run the anomaly detection procedure (Autoencoder), classify its detected anomalies to threats and publish the results to the Remediation Engine and to the Dashboard:
     ```sh
     $ ./sim_anom_class_pub.sh <YYYYMMDD or today>
    ```
    e.g:
    ```sh
     $ ./sim_anom_class_pub.sh 20181026
    ```
- To query the Hive Database for netflow logs of a specific date, classify them to threats and publish the results to the Remediation Engine and to the Dashboard:
     ```sh
     $ ./b_sim_anom_class_pub.sh <YYYYMMDD or today>
    ```
    e.g:
    ```sh
     $ ./b_sim_anom_class_pub.sh today
    ```




### Todos

 - Facilitate path configuration

License
----
Apache2


[![Funded by EU](https://www.shield-h2020.eu/shield-h2020/img/EC-H2020.png)](https://www.shield-h2020.eu/shield-h2020/img/EC-H2020.png)
