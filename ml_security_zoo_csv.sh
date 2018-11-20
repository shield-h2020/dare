#!/bin/bash

# parse and validate arguments

PHASE=$1
DSOURCE=$2
DPATH=$3
INPUT_TYPE=$4


if [[  -z "${PHASE}" || -z "${DSOURCE}" ]] ; then
    echo "ml_security.sh syntax error"
    echo "Please run ml_security.sh again with the correct syntax:"
    echo "./ml_security.sh PHASE TYPE YYYYMMDD"
    echo "for example:"
    echo "./ml_security.sh train flow 20160122"
    echo "./ml_security.sh test flow 20160122"
    exit
fi

# read in variables (except for date) from etc/.conf file
# note: FDATE and DSOURCE *must* be defined prior sourcing this conf file

source ./ml.conf

# pass the user domain designation if not empty

if [ ! -z $USER_DOMAIN ] ; then
    USER_DOMAIN_CMD="--userdomain $USER_DOMAIN"
else
    USER_DOMAIN_CMD=''
fi

FEEDBACK_PATH=${HPATH}/feedback/ml_feedback.csv

if [ "$PHASE" == "train" ]; then
    RAWDATA_PATH=${TESTING_PATH}
    HDFS_SCORED_CONNECTS=${TRAINING_PATH}
else
    RAWDATA_PATH=${TESTING_PATH}
    HDFS_SCORED_CONNECTS=${HPATH}/scores
fi

CONTAMINATION=0.03
NUMBER_OUTLIERS=100

ANALYTICS_ZOO_ROOT=./dependencies
ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/zoo
ANALYTICS_ZOO_PY_ZIP=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.6.0-spark_2.2.0-0.2.0-python-api.zip
ANALYTICS_ZOO_JAR=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.6.0-spark_2.2.0-0.2.0-jar-with-dependencies.jar
ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
PYTHONPATH=${ANALYTICS_ZOO_PY_ZIP}:$PYTHONPATH
VENV_HOME=${ANALYTICS_ZOO_HOME}/bin

time spark2-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python \
    --conf spark.yarn.maxAppAttempts=1 \
    --master yarn \
    --deploy-mode cluster\
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 2 \
    --num-executors 8 \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --jars ${ANALYTICS_ZOO_JAR} \
    --py-files ${ANALYTICS_ZOO_PY_ZIP},transformers_deep_learning.py \
    --archives ${VENV_HOME}/venv.zip \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    deep_learning_zoo.py \
    --mode ${PHASE} \
    --input ${DPATH} \
    --output ${HDFS_SCORED_CONNECTS} \
    --network ${TRAINING_PATH} \
    --contamination ${CONTAMINATION} \
    --number_outliers ${NUMBER_OUTLIERS} \
    --input_type ${INPUT_TYPE}
