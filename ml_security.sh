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
#

# parse and validate arguments

PHASE=$1
DSOURCE=$2

if [ "$PHASE" == "test" ]; then
    FDATE=$3
    YR=${FDATE:0:4}
    MH=${FDATE:4:2}
    DY=${FDATE:6:2}
fi


if [[ "$PHASE" == "train" && -z "${DSOURCE}" ]]; then
    echo "ml_security.sh syntax error"
    echo "Please run ml_security.sh again with the correct syntax:"
    echo "./ml_security.sh PHASE TYPE"
    echo "for example:"
    echo "./ml_security.sh train flow"
    exit
elif [[ "$PHASE" == "test"  && ( -z "${DSOURCE}" || "${#FDATE}" != "8" )]]; then
    echo "ml_security.sh syntax error"
    echo "Please run ml_security.sh again with the correct syntax:"
    echo "./ml_security.sh PHASE TYPE YYYYMMDD"
    echo "for example:"
    echo "./ml_security.sh test flow 20160122"
    echo "./ml_security.sh test proxy 20160122"
    exit
fi

# read in variables (except for date) from etc/.conf file
# note: FDATE and DSOURCE *must* be defined prior sourcing this conf file

source /home/spotuser/ml_security/ml.conf

# pass the user domain designation if not empty

if [ ! -z $USER_DOMAIN ] ; then
    USER_DOMAIN_CMD="--userdomain $USER_DOMAIN"
else
    USER_DOMAIN_CMD=''
fi

FEEDBACK_PATH=${HPATH}/feedback/ml_feedback.csv

if [ "$PHASE" == "train" ]; then
    RAWDATA_PATH=${TRAINING_PATH}
    HDFS_SCORED_CONNECTS=${TRAINING_PATH}/algorithms
else
    RAWDATA_PATH=${TESTING_PATH}
    HDFS_SCORED_CONNECTS=${HPATH}/scores
fi

if [ "$ALGORITHM" == "OneClassSVM" ]; then
  PARAMETERS=(${NU} ${KERNEL})
elif [ "$ALGORITHM" == "IsolationForest" ]; then
  PARAMETERS=(${ESTIMATORS} ${CONTAMINATION})
elif [ "$ALGORITHM" == "LocalOutlier" ]; then
  PARAMETERS=(${NEIGHBORS} ${CONTAMINATION})
fi


time spark2-submit \
  --master yarn \
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
  --conf spark.yarn.executor.memoryOverhead=${SPK_EXEC_MEM_OVERHEAD} \
  --conf spark.debug.maxToStringFields=100 \
  machine_learning.py \
  ${PHASE} \
  ${ALGORITHM} \
  ${PARAMETERS[0]} \
  ${PARAMETERS[1]} \
  --input ${RAWDATA_PATH} \
  --output ${HDFS_SCORED_CONNECTS} \
  --network ${TRAINING_PATH}/algorithms
