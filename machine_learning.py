#!/usr/bin/python

"""
Usage:
  machine_learning.py (train | test) OneClassSVM <nu> <kernel> options
  machine_learning.py (train | test) IsolationForest <estimator> <contam> options
  machine_learning.py (train | test) LocalOutlier <neigh> <contam> options

Modes:
  train   Execute the training phase
  test    Execute the testint phase

Options:
  -i <path>, --input <path>           # Netflow traffic input file path
  -o <path>, --output <path>          # Output file path
  -n <path>, --network <path>         # Path to the trained network file
"""

import sys
import pandas
import json
import numpy as np
import subprocess
import os
import transformers # Change value and divider

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import *

from pyspark.ml.feature import StringIndexer
from pyspark.mllib.feature import  StandardScaler
from pyspark.mllib.linalg import Vectors

from docopt import docopt, DocoptExit
from schema import Schema, SchemaError, And, Use, Or

from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib

# File name for the minimum-maximum values in the local filesystem
minMaxFile = "trained_min_max.json"

list_prots = ['UDP', 'TCP', 'ICMP', 'ICMP', 'IGMP']
list_flags = ['A', 'S', 'F', 'R', 'P', 'U', 'X']

def transform_protocol(proto):
    output_list = [0, 0, 0, 0, 0]

    for i in range(len(list_prots)):
        if proto in list_prots[i]:
            break

    output_list[i] += 1
    return output_list

def process_flag(flag):
    flag_array = [0, 0, 0, 0, 0, 0]
    contains_x = False

    for i in range(len(list_flags)):
        for letter in flag:
            if letter == list_flags[i]:
                flag_array[i] += 1

            if letter == 'X':
                contains_x = True

    if contains_x:
        return [1, 1, 1, 1, 1, 1]
    else:
        return flag_array

def NormalizeValues(df, arguments, json_min_max, logger):
    inputFeatures = ["tdur", "ipkt", "ibyt", "nconnections"]
    outputFeatures = ["tdNorm","ipktNorm","ibytNorm", "nconnNorm"]
    i = 0

    for feature in inputFeatures:
        logger.info("Normalizing {0} Column".format(feature))
        minDict = "min(" + feature + ")"
        maxDict = "max(" + feature + ")"

        if arguments['train'] == True:
            # Search the minimum and the maximum value from all the data in the feature column
            min = df.select(f.min(feature)).collect()[0].asDict()[minDict]
            max = df.select(f.max(feature)).collect()[0].asDict()[maxDict]

            # Add minimum and maximum to json file
            json_min_max['inputFeatures'].append({
                feature: [{
                    "min": min,
                    "max": max
                }]
            })

        elif arguments['test'] == True:
            min = json_min_max['inputFeatures'][i][feature][0]['min']
            max = json_min_max['inputFeatures'][i][feature][0]['max']

        std = max - min

        df = df.withColumn(outputFeatures[i], (df[feature] - min)/std)
        df = df.drop(feature)
        i += 1

    logger.info("Data values have been succesfully normalized.")
    return df, json_min_max

def preprocess(df, logger):

    logger.info("Starting preprocess analysis.")

    if df.rdd.isEmpty():
        logger.error("Couldn't read data")
        System.exit(0)

    # Change the columns name
    df =  df.withColumnRenamed('ts', 'treceived') \
            .withColumnRenamed('td', 'tdur') \
            .withColumnRenamed('sa', 'sip') \
            .withColumnRenamed('da', 'dip') \
            .withColumnRenamed('sp', 'sport') \
            .withColumnRenamed('dp', 'dport') \
            .withColumnRenamed('pr', 'proto') \
            .withColumnRenamed('flg', 'flag')

    # Add splited date fields
    if not 'tryear' in df.columns:
        split_column = f.split(df.treceived, ' ')
        df_date = split_column.getItem(0)
        df_time = split_column.getItem(1)
        df = df.withColumn("tryear", f.split(df_date, "-")[0])\
               .withColumn("trmonth", f.split(df_date, "-")[1])\
               .withColumn("trday", f.split(df_date, "-")[2])\
               .withColumn("trhour", f.split(df_time, ":")[0])\
               .withColumn("trminute", f.split(df_time, ":")[1])\
               .withColumn("trsec", f.split(df_time, ":")[2])

    if not 'opkt' in df.columns:
        df = df.withColumn("opkt", f.lit(0))\
               .withColumn("obyt", f.lit(0))

    # Select only the needed columns
    df = df.select(df.treceived,
                   df.tryear.cast(IntegerType()),
                   df.trmonth.cast(IntegerType()),
                   df.trday.cast(IntegerType()),
                   df.trhour.cast(IntegerType()),
                   df.trminute.cast(IntegerType()),
                   df.trsec.cast(IntegerType()),
                   df.tdur.cast(FloatType()),
                   df.sip,
                   df.dip,
                   df.sport.cast(IntegerType()),
                   df.dport.cast(IntegerType()),
                   df.proto,
                   df.flag,
                   df.ipkt.cast(FloatType()),
                   df.ibyt.cast(FloatType()),
                   df.opkt.cast(FloatType()),
                   df.obyt.cast(FloatType()))


    # Drop null values
    df = df.dropna(how='any')

    # Convert flow duration units from miliseconds to seconds
    df = df.withColumn("tdur", df.tdur/1000)

    # Preprocess the data by grouping the ts(start time), sa(source address), dp(destination port), pr(protocol) and flg(TCP flags) colums
    # Apply different operations to the td(flow duration), ipkt(number of input packets), ibyt(number of input bytes), pps(packets per second)
    # bps(bytes per second) and bpp (bytes per packets) colums
    df_sum = df.groupBy("treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","sport","dport", "proto", "flag").agg(f.sum(df.tdur),f.sum(df.ipkt),f.sum(df.ibyt),f.sum(df.opkt),f.sum(df.obyt))
    # Count the number of flows with the same features and add it in another column
    df_con = df.groupBy("treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","sport","dport", "proto","flag").count()
    # Add both results in a single DataFrame
    df_join = df_sum.join(df_con, [df_sum.treceived == df_con.treceived, df_sum.sip == df_con.sip, df_sum.dip == df_con.dip, df_sum.sport == df_con.sport, df_sum.dport == df_con.dport, df_sum.proto == df_con.proto, df_sum.flag == df_con.flag]).select(df_sum.treceived, df_sum.tryear, df_sum.trmonth, df_sum.trday, df_sum.trhour, df_sum.trminute, df_sum.trsec, df_sum.sip, df_sum.dip, df_sum.sport, df_sum.dport, df_sum.proto, df_sum.flag, df_sum["sum(tdur)"], df_sum["sum(ipkt)"], df_sum["sum(ibyt)"], df_sum["sum(opkt)"], df_sum["sum(obyt)"], df_con["count"])

    # Change the columns name
    df = df_join.withColumnRenamed("sum(tdur)", "tdur")\
                .withColumnRenamed("sum(ipkt)", "ipkt")\
                .withColumnRenamed("sum(ibyt)", "ibyt")\
                .withColumnRenamed("sum(opkt)", "opkt")\
                .withColumnRenamed("sum(obyt)", "obyt")\
                .withColumnRenamed('count', 'nconnections')

    # Apply a logarithmic transformation to the td, ipkt and ibyt columns
    logger.info('Applying logarithmic transform to columns')
    df = df.withColumn('tdur', f.log(df.tdur + 1)).withColumn('ipkt', f.log(df.ipkt + 1)).withColumn('ibyt', f.log(df.ibyt + 1))

    proto_transform = f.udf(lambda z: transform_protocol(z), ArrayType(IntegerType()))
    flag_transform = f.udf(lambda z: process_flag(z), ArrayType(IntegerType()))

    # Transform protocol column into one-hot encoding
    logger.info('Transforming protocol into one-hot encoding')
    df = df.withColumn('proto_onehot', proto_transform(df.proto))
    df = df.withColumn('proto_onehot0', df.proto_onehot[0])
    df = df.withColumn('proto_onehot1', df.proto_onehot[1])
    df = df.withColumn('proto_onehot2', df.proto_onehot[2])
    df = df.withColumn('proto_onehot3', df.proto_onehot[3])
    df = df.withColumn('proto_onehot4', df.proto_onehot[4])

    # Decode flag column and transform it into one-hot encoding
    logger.info('Transforming flag into one-hot encoding')
    df = df.withColumn('flag_onehot', flag_transform(df.flag))
    df = df.withColumn('flag_onehot0', df.flag_onehot[0])
    df = df.withColumn('flag_onehot1', df.flag_onehot[1])
    df = df.withColumn('flag_onehot2', df.flag_onehot[2])
    df = df.withColumn('flag_onehot3', df.flag_onehot[3])
    df = df.withColumn('flag_onehot4', df.flag_onehot[4])
    df = df.withColumn('flag_onehot5', df.flag_onehot[5])

    df = df.drop('flag_onehot', 'proto_onehot')

    return df

def training(sparkSession, arguments, logger):

    # Get the input file path
    inputPath = arguments['--input']
    logger.info("...Starting training...")
    logger.info("Loading data from: {0}".format(inputPath))

    # Read the input dataset
    trainDF = sparkSession.read.parquet(inputPath)
    # Preprocess the data
    processData = preprocess(trainDF, logger)

    # Select final columns for training the algorithms
    processData = processData.select(processData.tdur.cast(FloatType()),
                                     processData.sport.cast(IntegerType()),
                                     processData.dport.cast(IntegerType()),
                                     processData.flag_onehot0,
                                     processData.flag_onehot1,
                                     processData.flag_onehot2,
                                     processData.flag_onehot3,
                                     processData.flag_onehot4,
                                     processData.flag_onehot5,
                                     processData.proto_onehot0,
                                     processData.proto_onehot1,
                                     processData.proto_onehot2,
                                     processData.proto_onehot3,
                                     processData.proto_onehot4,
                                     processData.ipkt.cast(FloatType()),
                                     processData.ibyt.cast(FloatType()),
                                     processData.opkt.cast(FloatType()),
                                     processData.obyt.cast(FloatType()),
                                     processData.nconnections.cast(IntegerType()))
    ## Normalize the data
    # Initialize a dictionary for the minimum and the maximum values for each normalized feature
    min_max = {}
    min_max['inputFeatures'] = []

    [dataNor, min_max] = NormalizeValues(processData, arguments, min_max, logger)

    # Save the minimum-maximum json file in the local filesystem
    with open(minMaxFile,"w") as f:
        f.write(json.dumps(min_max))

    # Transform Spark Data Frame into Pandas Data Frame
    dataNor = dataNor.toPandas()
    # Get the values to train the network
    dataTrain = dataNor.values

    # Select the training algorithm
    if arguments['OneClassSVM'] == True:
        nu = arguments['<nu>']
        kernel = arguments['<kernel>']
        alg = svm.OneClassSVM(kernel=kernel, nu=nu)
        algorithm = "OneClassSVM"
        logger.info("One Class SVM model")
    elif arguments['IsolationForest'] == True:
        estimators = arguments['<estimator>']
        contamination = arguments['<contam>']
        logger.info("Estimators: {0}; Contamination: {1}".format(estimators, contamination))
        alg = IsolationForest(n_estimators=estimators, contamination=contamination)
        algorithm = "IsolationForest"
        logger.info("Isolation model")
    elif arguments['LocalOutlier'] == True:
        neighbors = arguments['<neigh>']
        contamination = arguments['<contam>']
        alg = LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination)
        algorithm = "LocalOutlier"
        logger.info("Local Outlier Factor model")

    logger.info("Fitting the network")
    # Training the network with the data
    alg.fit(dataTrain)
    logger.info("Algorithm has been trained.")

    algFile = algorithm + "_network.plk"
    # Copy the trained network to the network_trained parameter
    joblib.dump(alg, algFile)

    # Get the path to save the results
    hdfsTrainDir = arguments['--output']
    logger.info("Saving results to {0}".format(hdfsTrainDir))

    # Get HDFS structures
    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    hdfsAlgPath = hdfsTrainDir + "/algorithms/" + algorithm + "_network.plk"
    hdfsMinMaxPath = hdfsTrainDir + "/" +  minMaxFile
    if(fs.exists(path(hdfsAlgPath)) == True):
        logger.warn("It already exists a trained network for the {0} algorithm in {1}".format(algorithm, hdfsTrainDir))
        logger.warn("The file is going to be overrited.")
        try:
          fs.delete(path(hdfsAlgPath), False)
        except:
          logger.error("Couldn't delete the network file")

    if(fs.exists(path(hdfsMinMaxPath)) == True):
        logger.warn("It already exists a file with the minimum and maximum values in {0}".format(hdfsTrainDir))
        logger.warn("The file is going to be overrited.")
        try:
          fs.delete(path(hdfsMinMaxPath), False)
        except:
          logger.error("Couldn't delete the minimum and maximum file")

    try:
         srcAlgFile = path(algFile)
         dstAlgFile = path(hdfsAlgPath)
         fs.moveFromLocalFile(srcAlgFile, dstAlgFile)
         logger.info("Training model exported correctly.")
    except:
         logger.error("Couldn't save the network in the file system.")

    try:
         srcMinMaxFile = path(minMaxFile)
         dstMinMaxFile = path(hdfsMinMaxPath)
         fs.moveFromLocalFile(srcMinMaxFile, dstMinMaxFile)
         logger.info("Minimum and maximum features file exported correctly.")
    except:
         logger.error("Couldn't save the minimum and maximum file in the file system.")

    logger.info("..Training has finished..")

def testing(sparkSession, arguments, logger):

    inputPath = arguments['--input']
    logger.info("...Starting testing...")
    logger.info("Loading data from: {0}".format(inputPath))

    # Get the path to get the minimum and maximum values from the training stage and to save the results
    hdfsTrainDir = arguments['--network']

    ## Load minimum and maximum values obtained in the training stage
    # HDFS filesystem path
    hdfsMinMaxPath = hdfsTrainDir + "/" +  minMaxFile

    # Get HDFS structures
    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fileUtil = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileUtil
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    if(fs.exists(path(hdfsMinMaxPath)) == True):
        logger.info("Getting the minimum and maximum file from: {0}".format(hdfsTrainDir))
        srcFile = path(hdfsMinMaxPath)
        dstFile = path(minMaxFile)
        fs.copyToLocalFile(False, srcFile, dstFile)
    else:
        logger.error("Couldn't find results in {0}".format(hdfsTrainDir))
        sys.exit(0)

    # Get the file from local filesystem
    json_max_min = json.loads(open(minMaxFile).read())
    # Remove the file from filesystem
    os.remove(minMaxFile)
    os.remove("." + minMaxFile + ".crc")

    # Read the dataset
    testDF = sparkSession.read.parquet(inputPath)
    # Preprocess the data
    processData = preprocess(testDF, logger)
    #Normalize the data
    [dataNor, json_max_min]= NormalizeValues(processData, arguments, json_max_min, logger)
    # Transform Spark Data Frame into Pandas Data Frame
    dataNor = dataNor.toPandas()
    dataNor = dataNor.set_index(["treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","proto","flag"])

    #Get the values to train the network
    dataTest = dataNor.values
    dataNor = dataNor.reset_index()

    ## Load the trained network obtained into the training phase
    # Select the training algorithm
    if arguments['OneClassSVM'] == True:
        algorithm = "OneClassSVM"
    elif arguments['IsolationForest'] == True:
        algorithm = "IsolationForest"
    elif arguments['LocalOutlier'] == True:
        algorithm = "LocalOutlier"

    algFile = algorithm + "_network.plk"
    hdfsAlgPath = hdfsTrainDir + "/algorithms/" + algorithm +  "_network.plk"

    if(fs.exists(path(hdfsAlgPath)) == True):
        logger.info("Getting the network from: {0}".format(hdfsTrainDir))
        srcFile = path(hdfsAlgPath)
        dstFile = path(algFile)
        fs.copyToLocalFile(False, srcFile, dstFile)
    else:
        logger.error("Couldn't find results in {0}".format(hdfsTrainDir))
        sys.exit(0)

    # Get the network from local filesystem
    alg = joblib.load(algFile)

    # Remove the network from filesystem
    os.remove(algFile)
    os.remove("." + algFile + ".crc")

    # Predict the results with the new data
    if arguments['LocalOutlier'] == True:
        prediction = alg.fit_predict(dataTest)
        hits = alg._decision_function(dataTest)
    elif ((arguments['OneClassSVM'] == True) or (arguments['IsolationForest'] == True)):
        prediction = alg.predict(dataTest)
        hits = alg.decision_function(dataTest)

    dataNor["pred"] = prediction
    dataNor["score"] = hits

    # Convert Data Frame into Spark Data Frame
    dataNor = sparkSession.createDataFrame(dataNor).drop("opkt","obyt")
    processData = processData.join(dataNor, ["treceived","tryear", "trmonth", "trday", "trhour", \
                                            "trminute", "trsec","sip","dip","sport","dport","proto", "flag"])\
                             .drop("tdNorm","ipktNorm","ibytNorm","nconnNorm","proto_onehot0","proto_onehot1",\
                             "proto_onehot2","proto_onehot3","proto_onehot4","flag_onehot0","flag_onehot0", \
                             "flag_onehot1", "flag_onehot2", "flag_onehot3", "flag_onehot4", "flag_onehot5")

    # Get the anomalies from the process Data
    outliers = processData.filter(processData.pred == -1)
    outliers = outliers.orderBy("score", ascending=True)
    # Order the columns
    outliers = outliers.select(outliers.treceived,
                                     outliers.tryear,
                                     outliers.trmonth,
                                     outliers.trday,
                                     outliers.trhour,
                                     outliers.trminute,
                                     outliers.trsec,
                                     outliers.tdur,
                                     outliers.sip,
                                     outliers.dip,
                                     outliers.sport,
                                     outliers.dport,
                                     outliers.proto,
                                     outliers.ipkt.cast(IntegerType()),
                                     outliers.ibyt.cast(IntegerType()),
                                     outliers.opkt.cast(IntegerType()),
                                     outliers.obyt.cast(IntegerType()),
                                     outliers.score)

    # Get the path to save the results
    hdfsScoredDir = arguments['--output']
    logger.info("Saving results in {0}".format(hdfsScoredDir))

    #Save the Data Frame into a CSV file
    if(fs.exists(path(hdfsScoredDir)) == False):
        logger.info("Saving results in {0}".format(hdfsScoredDir))
        fs.mkdirs(path(hdfsScoredDir))
        logger.info("The path {0} has been created.".format(hdfsScoredDir))

    hdfsScoredPath = hdfsScoredDir + "/" + "flow_results.csv"
    if(fs.exists(path(hdfsScoredPath)) == True):
        logger.warn("It already exists a results file in {0}".format(hdfsScoredDir))
        logger.warn("The file is going to be overrited.")
        try:
          fs.delete(path(hdfsScoredPath), False)
        except:
          logger.error("Couldn't delete the file")
    try:
        outliers.write.csv(hdfsScoredDir + "/anomalyResults", mode="append", sep="\t")
        srcPath = path(hdfsScoredDir + "/anomalyResults")
        dstPath = path(hdfsScoredPath)
        fileUtil.copyMerge(fs, srcPath, fs, dstPath, True, fs.getConf(), "")
        logger.info("The data has been succesfully saved into HDFS filesystem")
    except:
        logger.error("Couldn't merge the files.")

    logger.info("..Testing has finished..")

if __name__ == "__main__":

    try:

        arguments = docopt(__doc__)
        schema = Schema({'<nu>': Or(None, And(Use(float), lambda n: 0 < n <= 1, error='NU should be float 0 < NU <= 1')),
                         '<kernel>': Or(None, And(Use(str), lambda s: s in ('linear', 'poly', 'sigmoid', 'rbf', 'precomputed'), error='KERNEL should be string ("linear","poly","sigmoid","rbf","precomputed")')),
                         '<estimator>': Or(None, And(Use(int), error='ESTIMATORS should be integer')),
                         '<neigh>': Or(None, And(Use(int), error='NEIGHBORS should be integer')),
                         '<contam>': Or(None, And(Use(float), lambda n: 0 < n <= 0.5, error='CONTAMINATION should be float 0 < CONTAMINATION <= 0.5')),
                         str: object})
        try:
            arguments = schema.validate(arguments)
            sparkSession = SparkSession\
                           .builder\
                           .appName("Security ML")\
                           .getOrCreate()

            Level = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.Level
            LogManager = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.LogManager
            Logger = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.Logger

            logger = LogManager.getLogger("MachineLearningSecurity")
            logger.setLevel(Level.INFO)

            Logger.getLogger("org").setLevel(Level.OFF)
            Logger.getLogger("akka").setLevel(Level.OFF)

            if arguments['train'] == True:
                training(sparkSession, arguments, logger)
            elif arguments['test'] == True:
                testing(sparkSession, arguments, logger)

            sparkSession.stop()
        except SchemaError as e:
            print e.message

    except DocoptExit as e:
        print e.message
