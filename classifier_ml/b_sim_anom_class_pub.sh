
#!/bin/bash

if [ $1 = "today" ]; then
   DATE=`date +%Y%m%d`
else
   DATE=$1
fi


YEA=${DATE:0:4}
MON=${DATE:4:2}
DAY=${DATE:6:2}

echo "$(tput setab 2)Fetching Hive data for $DATE $(tput sgr 0)"


hive -hiveconf yea="$YEA" -hiveconf mon="$MON" -hiveconf day="$DAY" -e ' select * from spot.flow  where (y="${hiveconf:yea}" AND m="${hiveconf:mon}" AND d="${hiveconf:day}")'  > flow_results.csv

echo "$(tput setab 2) Removing data from previous runs $(tput sgr 0)"

#PATH OF THE HDFS FOLDER OF PREVIOUSLY SAVED ANOMALY RESULTS (REMOVING THEM)
hdfs dfs -rm /user/spotuser/rf/flow_results.csv


echo "$(tput setab 2)Loading data to HDFS for threat identification $(tput sgr 0)"
#THE FOLDER GETS THE FILE THAT WAS GENERATED BY THE HIVE QUERY
hdfs dfs -copyFromLocal ~/flow_results.csv /user/rf/testfolder/

echo "$(tput setab 2)Starting threat identification for: $DATE.$(tput sgr 0)"

#LOCAL PATH OF THE RANDOM FOREST CLASSIFIER.
cd /home/spotuser/Demo_RF_Classifier/demo_rf_classifier/

./classifier3.sh predict  /user/spotuser/rf/flow_results.csv

./publish.sh


