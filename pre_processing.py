from __future__ import print_function, division, absolute_import

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql import functions as F

import argparse

import sys


list_prots = ['UDP', 'TCP', 'ICMP', 'ICMP', 'IGMP']


def transform_protocol(proto):
    output_list = [0, 0, 0, 0, 0]

    for i in range(len(list_prots)):
        if proto in list_prots[i]:
            break

    output_list[i] += 1
    return output_list


list_flags = ['A', 'S', 'F', 'R', 'P', 'U', 'X']


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


def preprocess(df, logger):
    # Check if database is empty
    if df.rdd.isEmpty():
        logger.error("Couldn't read data")
        sys.exit(0)

    # Select interesting columns (the only ones containing data)
    logger.info('Selecting interesting columns')
    df = df.select(df.td.cast('float'),
                   df.sp.cast('int'),
                   df.dp.cast('int'),
                   df.pr,
                   df.flg,
                   df.ipkt.cast('int'),
                   df.ibyt.cast('int'))

    # Remove rows with NaN values
    df = df.dropna()

    # Apply a logarithmic transformation to the td, ipkt and ibyt columns
    logger.info('Applying logarithmic transorm to columns')
    df = df.withColumn('td', F.log(df.td + 1)).withColumn('ipkt', F.log(df.ipkt + 1)).withColumn('ibyt',
                                                                                                 F.log(df.ibyt + 1))

    proto_transform = F.udf(lambda z: transform_protocol(z), ArrayType(IntegerType()))

    logger.info('Transforming flag into one-hot encoding')
    flag_transform = F.udf(lambda z: process_flag(z), ArrayType(IntegerType()))


    # Transform protocol column into one-hot encoding
    logger.info('Transforming protocol into one-hot encoding')
    df = df.withColumn('proto_onehot', proto_transform(df.pr))
    df = df.withColumn('proto_onehot0', df.proto_onehot[0])
    df = df.withColumn('proto_onehot1', df.proto_onehot[1])
    df = df.withColumn('proto_onehot2', df.proto_onehot[2])
    df = df.withColumn('proto_onehot3', df.proto_onehot[3])
    df = df.withColumn('proto_onehot4', df.proto_onehot[4])

    # Decode flag column and transform it into one-hot encoding
    logger.info('Transforming flag into one-hot encoding')
    df = df.withColumn('flag_onehot', flag_transform(df.flg))
    df = df.withColumn('flag_onehot0', df.flag_onehot[0])
    df = df.withColumn('flag_onehot1', df.flag_onehot[1])
    df = df.withColumn('flag_onehot2', df.flag_onehot[2])
    df = df.withColumn('flag_onehot3', df.flag_onehot[3])
    df = df.withColumn('flag_onehot4', df.flag_onehot[4])
    df = df.withColumn('flag_onehot5', df.flag_onehot[5])


    # Select final columns for training the algorithms
    df = df.select(df.td.cast('float'),
                   df.flag_onehot0,
                   df.flag_onehot1,
                   df.flag_onehot2,
                   df.flag_onehot3,
                   df.flag_onehot4,
                   df.flag_onehot5,
                   df.proto_onehot0,
                   df.proto_onehot1,
                   df.proto_onehot2,
                   df.proto_onehot3,
                   df.proto_onehot4,
                   df.ipkt.cast('float'),
                   df.ibyt.cast('float'))

    return df


feat_to_norm = ["td", "ipkt", "ibyt"]


def normalize(train_df, test_df, logger):
    # Obtain the mean and std values from the train database and use them to normalize train and test
    for feat in feat_to_norm:
        mean = train_df.select(F.avg(feat)).collect()[0][0]
        std = train_df.select(F.stddev(feat)).collect()[0][0]

        train_df = train_df.withColumn(feat, (train_df[feat] - mean) / std)
        test_df = test_df.withColumn(feat, (test_df[feat] - mean) / std)

        logger.info(feat + ' normalized')

    return train_df, test_df


def main_fun(spark_session, args, logger):
    logger.info('Inside main fun')

    input_path_train = args.input_train
    input_path_test = args.input_test

    logger.info('Input path of the train CSV is ' + input_path_train)
    logger.info('Input path of the test CSV is ' + input_path_test)

    train_df = spark_session.read.format("csv").option("header", "true").load(input_path_train)
    test_df = spark_session.read.format("csv").option("header", "true").load(input_path_test)

    # Only for the MouseWorld database, it seems to have a lot of requests to this IP (seem pings to google)
    train_df = train_df.filter((train_df.da != '8.8.8.8'))
    test_df = test_df.filter((test_df.da != '8.8.8.8'))

    train_df = preprocess(train_df, logger)
    test_df = preprocess(test_df, logger)

    train_df, test_df = normalize(train_df, test_df, logger)

    # Do whatever with the datasets, here we save the numpy arrays to process them using sklearn and keras
    np.save('train_data.npy', train_df.toPandas().values)
    np.save('test_data.npy', test_df.toPandas().values)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_train', required=True, help='Patch for the train input csv')
    parser.add_argument('--input_test', required=True,  help='Patch for the test input csv')

    args = parser.parse_args()

    spark_session = SparkSession.builder.appName("Security ML").getOrCreate()

    Level = spark_session.sparkContext._gateway.jvm.org.apache.log4j.Level
    LogManager = spark_session.sparkContext._gateway.jvm.org.apache.log4j.LogManager
    Logger = spark_session.sparkContext._gateway.jvm.org.apache.log4j.Logger

    logger = LogManager.getLogger("MachineLearningSecurity")
    logger.setLevel(Level.INFO)

    main_fun(spark_session, args, logger)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    spark_session.stop()
