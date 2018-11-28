#!/bin/bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


PHASE=$1
SOURCE=$2
TREES=$3

#LOCAL PATH OF THE ML.CONF FILE, NEEDED FOR THE CLASSIFIER
source /home/spotuser/Demo_RF_Classifier_worm/ml.conf



if [[ "$PHASE" == "train" ]] ; then
    
    if [ !-z "$SOURCE" ]; then    
    
        echo "Training phase selected. Starting Spark job."

        time spark2-submit \
        --driver-memory 3g \
        --conf spark.executor.memory=8g \
        --conf spark.driver.maxResultSize=3g \
        --conf spark.driver.maxPermSize=3g \
        --conf spark.dynamicAllocation.enabled=true \
        --conf spark.dynamicAllocation.maxExecutors=-1 \
        --conf spark.executor.cores=-1 \
        --conf spark.executor.memory=3g \
        --conf spark.sql.autoBroadcastJoinThreshold=-1 \
        --conf "spark.executor.extraJavaOptions=-XX:MaxPermSize=512M -XX:PermSize=512M -verbose:gc -XX:+PrintGCDetails" \
        --conf spark.kryoserializer.buffer.max=512m \
        --conf spark.yarn.am.waitTime=100s \
        --conf spark.yarn.am.memoryOverhead=50 \
        --conf spark.executor.memoryOverhead=50 \
        --conf spark.debug.maxToStringFields=100 \
        threat_classifier0.4.py -p ${PHASE} -c ${SOURCE} -t ${TREES} 2>classtrainlogs.out
    
    else echo "File doesn't exist"
    
    fi
    

elif [[ "$PHASE" == "predict" ]] ; then

    if [ ! -z "$SOURCE" ]; then
    
    echo "Classification phase selected. Starting Spark job."

    time spark2-submit \
    --driver-memory ${SPK_DRIVER_MEM} \
    --conf spark.executor.memory=8g \
    --conf spark.driver.maxResultSize=${SPK_DRIVER_MAX_RESULTS} \
    --conf spark.driver.maxPermSize=512m \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.maxExecutors=${SPK_EXEC} \
    --conf spark.executor.cores=${SPK_EXEC_CORES} \
    --conf spark.executor.memory=${SPK_EXEC_MEM} \
    --conf spark.sql.autoBroadcastJoinThreshold=${SPK_AUTO_BRDCST_JOIN_THR} \
    --conf "spark.executor.extraJavaOptions=-XX:MaxPermSize=512M -XX:PermSize=512M -verbose:gc -XX:+PrintGCDetails" \
    --conf spark.kryoserializer.buffer.max=512m \
    --conf spark.yarn.am.waitTime=100s \
    --conf spark.yarn.am.memoryOverhead=${SPK_DRIVER_MEM_OVERHEAD} \
    --conf spark.yarn.memoryOverhead=${SPK_EXEC_MEM_OVERHEAD} \
    --conf spark.debug.maxToStringFields=100 \
    threat_classifier0.4.py -p ${PHASE} -c ${SOURCE}  2>classpredictlogs.out

    else echo "File doesn't exist"
    
    fi

elif [[ "$PHASE" == "help" || "$PHASE" == "-h" ]]; then

    echo "./classifier3.sh [train|test] [filepath] [treenumber (optional)]"
    echo "for example:"
    echo "./classifier3.sh train train.csv  50"
    echo "or"
    echo "./classifier3.sh predict flow_results.csv"

else
    echo " Wrong syntax. Please run classifier.sh again with the correct syntax:"
    echo "./classifier3.sh [train|test] [filepath] [treenumber (optional)]"
    echo "for example:"
    echo "./classifier3.sh train train.csv  50"
    echo "or"
    echo "./classifier3.sh predict flow_results.csv"
    
    
fi  
