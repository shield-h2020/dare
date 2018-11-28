#!/bin/bash

if [ $1 = "today" ]; then
    DATE=`date +%Y%m%d`
else
    DATE=$1
fi

echo "$(tput setab 2) Starting anomaly detection for: $DATE $(tput sgr 0)"

# LOCAL PATH OF THE AUTOENCODER
cd /home/spotuser/demo_2018/ShieldDL/

/home/spotuser/demo_2018/ShieldDL/ml_security_zoo.sh test flow $DATE

echo "$(tput setab 2)Starting threat classification for: $DATE.$(tput sgr 0)"

# LOCAL PATH OF THE RANDOM FOREST CLASSIFIER
cd /home/spotuser/Demo_RF_Classifier_worm/

#PATH OF THE HDFS FOLDER WHERE ANOMALY DETECTION SAVES THE scored_results.
./classifier3.sh predict /user/spotuser/flow/scored_results/$DATE/scores/flow_results.csv

./publish.sh

 
