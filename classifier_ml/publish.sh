#!/bin/bash

source ~/Demo_RF_Classifier_worm/venv/bin/activate

CSV_FOLDER="/home/spotuser/Demo_RF_Classifier_worm/messages/"
CSV_FILE="/home/spotuser/Demo_RF_Classifier_worm/messages/message.csv"

echo "$(tput setab 2)Removing old transmissions...$(tput sgr 0)"
rm -f $CSV_FOLDER/*

echo "Fetching threat file from HDFS...$(tput sgr 0)"

hdfs dfs -copyToLocal /user/spotuser/rf/threatresults_flow_results.csv/part* $CSV_FOLDER
cat $CSV_FOLDER/*.csv > $CSV_FILE

first_line=$(head -n 1 $CSV_FILE)

CSV_ID=`echo $first_line | awk '{print $1}'`
SEVERITY=`echo $first_line | awk '{print $2}'`
ATTACK_TYPE=`echo $first_line | awk '{print $3}'`

echo "Starting transmission:$(tput sgr 0)"

python ~/Demo_RF_Classifier_worm/rmq_sender.py line "$CSV_ID	$SEVERITY	$ATTACK_TYPE	start"
sleep .2
cat  $CSV_FILE|head -30 | while read line
do
        python ~/Demo_RF_Classifier_worm/rmq_sender.py line "$line"
	sleep .01
done 
sleep .2
python ~/Demo_RF_Classifier_worm/rmq_sender.py line "$CSV_ID	$SEVERITY	$ATTACK_TYPE	stop"

echo "$(tput setab 2)Transmission completed.$(tput sgr 0)"

deactivate

 
