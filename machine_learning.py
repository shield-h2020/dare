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

def NormalizeValues(df, logger):
    inputFeatures = ["tdur", "ipkt", "ibyt", "pps", "bps", "bpp", "nconnections"]
    outputFeatures = ["tdNorm","ipktNorm","ibytNorm","ppsNorm","bpsNorm","bppNorm","nconnNorm"]
    i = 0
    for feature in inputFeatures:
        logger.info("Normalizing {0} Column".format(feature))
        minDict = "min(" + feature + ")"
        maxDict = "max(" + feature + ")"
        min = df.select(f.min(feature)).collect()[0].asDict()[minDict]
        max = df.select(f.max(feature)).collect()[0].asDict()[maxDict]
        std = max - min

        df = df.withColumn(outputFeatures[i], (df[feature] - min)/std)
        df = df.drop(feature)
        i += 1
    logger.info("Data values have been succesfully normalized.")
    return df

def preprocess(df, logger):

    logger.info("Starting preprocess analysis.")

    # Drop null values
    df = df.dropna(how='any')

    # Change the columns name
    df =  df.withColumnRenamed('ts', 'treceived') \
            .withColumnRenamed('td', 'tdur') \
            .withColumnRenamed('sa', 'sip') \
            .withColumnRenamed('da', 'dip') \
            .withColumnRenamed('sp', 'sport') \
            .withColumnRenamed('dp', 'dport') \
            .withColumnRenamed('pr', 'proto') \
            .withColumnRenamed('flg', 'flag')

    if df.rdd.isEmpty():
        logger.error("Couldn't read data")
        System.exit(0)

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

    # Convert flow duration units from miliseconds to seconds
    df = df.withColumn("tdur", df.tdur/1000)

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

    ## Define the list of transforms to apply to the data
    # Change the value td from 0.0 to 0.0005
    change_td = transformers.ChangeValue(column="tdur", value_initial=0.0000, value_change=0.0005)
    # Add packets per second
    insert_pps = transformers.Divider(inColumn1 = "ipkt", inColumn2 = "tdur", outColumn = "pps")
    # Add bytes per second
    insert_bps = transformers.Divider(inColumn1 = "ibyt", inColumn2 = "tdur", outColumn = "bps")
    # Add bytes per packet
    insert_bpp = transformers.Divider(inColumn1 = "ipkt", inColumn2 = "ibyt", outColumn = "bpp")
    # Encode protocol feature to numerical values
    encoder_pr = StringIndexer(inputCol = "proto", outputCol = "protoIndex", handleInvalid='error')
    # Encode TCP flags feature to numerical values
    encoder_flg = StringIndexer(inputCol = "flag", outputCol = "flagIndex", handleInvalid='error')

    # Apply the transformations
    df =  change_td.transform(df)
    logger.info("Filling empty time duration rows with 0.0005 value.")
    df =  insert_pps.transform(df)
    logger.info("Packets per second has been succesfully calculated.")
    df =  insert_bps.transform(df)
    logger.info("Bytes per second has been succesfully calculated.")
    df =  insert_bpp.transform(df)
    logger.info("Bytes per packet has been succesfully calculated.")
    pr_model = encoder_pr.fit(df)
    df = pr_model.transform(df)
    logger.info("Encoding protocol row has been done.")
    flg_model = encoder_flg.fit(df)
    df = flg_model.transform(df)
    logger.info("Encoding flag row has been done.")

    # Preprocess the data by grouping the ts(start time), sa(source address), dp(destination port), pr(protocol) and flg(TCP flags) colums
    # Apply different operations to the td(flow duration), ipkt(number of input packets), ibyt(number of input bytes), pps(packets per second)
    # bps(bytes per second) and bpp (bytes per packets) colums
    df_sum = df.groupBy("treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","sport","dport" , "proto","protoIndex", "flagIndex").agg(f.sum(df.tdur),f.sum(df.ipkt),f.sum(df.ibyt),f.sum(df.pps),f.sum(df.bps),f.sum(df.bpp),f.sum(df.opkt),f.sum(df.obyt))
    # Count the number of flows with the same features and add it in another column
    df_con = df.groupBy("treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","sport","dport" , "proto","protoIndex", "flagIndex").count()
    # Add both results in a single DataFrame
    df_join = df_sum.join(df_con, [df_sum.treceived == df_con.treceived, df_sum.sip == df_con.sip, df_sum.dip == df_con.dip, df_sum.sport == df_con.sport, df_sum.dport == df_con.dport, df_sum.proto == df_con.proto, df_sum.protoIndex == df_con.protoIndex, df_sum.flagIndex == df_con.flagIndex]).select(df_sum.treceived, df_sum.tryear, df_sum.trmonth, df_sum.trday, df_sum.trhour, df_sum.trminute, df_sum.trsec, df_sum.sip, df_sum.dip, df_sum.sport, df_sum.dport, df_sum.proto, df_sum.protoIndex, df_sum.flagIndex, df_sum["sum(tdur)"], df_sum["sum(ipkt)"], df_sum["sum(ibyt)"], df_sum["sum(pps)"], df_sum["sum(bps)"], df_sum["sum(bpp)"], df_sum["sum(opkt)"], df_sum["sum(obyt)"], df_con["count"])

    # Change the columns name
    df_join  =  df_join.withColumnRenamed("flagIndex", "flag")\
            .withColumnRenamed("sum(tdur)", "tdur")\
            .withColumnRenamed("sum(ipkt)", "ipkt")\
            .withColumnRenamed("sum(ibyt)", "ibyt")\
            .withColumnRenamed("sum(pps)", "pps")\
            .withColumnRenamed("sum(bps)", "bps")\
            .withColumnRenamed("sum(bpp)", "bpp")\
            .withColumnRenamed("sum(opkt)", "opkt")\
            .withColumnRenamed("sum(obyt)", "obyt")\
            .withColumnRenamed('count', 'nconnections')

    return df_join

def training(sparkSession, arguments, logger):

    inputPath = arguments['--input']
    logger.info("...Starting training...")
    logger.info("Loading data from: {0}".format(inputPath))

    # Read the clean dataset
    trainDF = sparkSession.read.format("csv").option("header", "true").load(inputPath)
    # Preprocess the data
    processData = preprocess(trainDF, logger)
    #Normalize the data
    dataNor= NormalizeValues(processData, logger)
    # Transform Spark Data Frame into Pandas Data Frame
    dataNor = dataNor.toPandas()
    #dataNor = dataNor.set_index('treceived')
    dataNor = dataNor.set_index(["treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip", "proto"])
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
    hdfsAlgDir = arguments['--output']
    logger.info("Saving results to {0}".format(hdfsAlgDir))

    # Get HDFS structures
    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    hdfsAlgPath = hdfsAlgDir + "/" + algorithm + "_network.plk"
    if(fs.exists(path(hdfsAlgPath)) == True):
        logger.warn("It already exists a trained network for the {0} algorithm in {1}".format(algorithm, hdfsAlgDir))
        logger.warn("The file is going to be overrited.")
        try:
          fs.delete(path(hdfsAlgPath), False)
        except:
          logger.error("Couldn't delete the network file")

    try:
         srcFile = path(algFile)
         dstFile = path(hdfsAlgPath)
         fs.moveFromLocalFile(srcFile, dstFile)
         logger.info("Training model exported correctly.")
    except:
         logger.error("Couldn't save the network in the file system.")
    logger.info("..Training has finished..")

def testing(sparkSession, arguments, logger):

    inputPath = arguments['--input']
    logger.info("...Starting testing...")
    logger.info("Loading data from: {0}".format(inputPath))

    # Read the dataset
    testDF = sparkSession.read.parquet(inputPath)
    # Preprocess the data
    processData = preprocess(testDF, logger)
    #Normalize the data
    dataNor= NormalizeValues(processData, logger)
    # Transform Spark Data Frame into Pandas Data Frame
    dataNor = dataNor.toPandas()
    dataNor = dataNor.set_index(["treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","proto"])
    dataNor = dataNor.fillna(0)
    # Get the values to train the network
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

    # Get the path to save the results
    hdfsAlgDir = arguments['--network']

    # Get HDFS structures
    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fileUtil = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileUtil
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    hdfsAlgPath = hdfsAlgDir + "/" + algorithm +  "_network.plk"
    if(fs.exists(path(hdfsAlgPath)) == True):
        logger.info("Getting the network from: {0}".format(hdfsAlgDir))
        srcFile = path(hdfsAlgPath)
        dstFile = path(algFile)
        fs.copyToLocalFile(False, srcFile, dstFile)
    else:
        logger.error("Couldn't find results in {0}".format(hdfsAlgDir))
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
    processData = processData.join(dataNor, ["treceived","tryear", "trmonth", "trday", "trhour", "trminute", "trsec","sip","dip","sport","dport","proto"]).drop("protoIndex","flag","pps","bps","bpp","tdNorm","ipktNorm","ibytNorm","ppsNorm","bpsNorm","bppNorm","nconnNorm")
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
        logger.info("Process Data Rows {0}".format(processData.count()))
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
