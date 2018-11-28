from pyspark.sql import SparkSession

import argparse

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

import json

from transformers_deep_learning import *
import sys


# columns_bad =  ['tr','try','trm','trd','trh','trm','trs','td','sa','da','sp','dp','pr','flg','fwd','stos','ipkt','ibyt','opkt','obyt','in','out','sas','das','dtos','dir','ra','classe']


def read_dataframe(sparkSession, args, logger):
    # Get the input file path
    inputPath = args.input
    logger.info("Loading data from: {0}".format(inputPath))

    if args.input_type == 'csv':
        logger.info('Reading from CSV')
        df = sparkSession.read.format('csv').option("header", "true").option("inferSchema", "true").load(inputPath)
    # Read the input dataset
    else:
        df = sparkSession.read.option("timestampFormat", "yyyy-MM-dd HH:mm:ss").parquet(inputPath)

    # This is only needed if data is coming from polito since they use a different version of nfcapd

    if 'td' in df.columns:
        logger.info('Columns are bad, renaming...')

        # Change the columns name to fit the ones in the original data types
        df = df.withColumnRenamed('td', 'tdur') \
            .withColumnRenamed('pr', 'proto') \
            .withColumnRenamed('flg', 'flag') \
            .withColumnRenamed('sp', 'sport') \
            .withColumnRenamed('dp', 'dport') \

        df = df.drop('trm')
        df = df.drop('trm.1')

    else:
        logger.info('Columns are good')


    if df.rdd.isEmpty():
        logger.error("Couldn't read data")
        sys.exit(0)

    # Remove rows with NaN values
    df = df.dropna(how='any')

    return df


def train(sample_rdd, epochs = 10, batch_size = 1024):

    model = Sequential()
    model.add(Linear(16, 12))
    model.add(Tanh())
    model.add(Linear(12, 16))

    # Create an Optimizer
    optimizer = Optimizer(
        model=model,
        training_rdd=sample_rdd,
        criterion=MSECriterion(),
        optim_method=Adam(learningrate=0.001),
        end_trigger=MaxEpoch(epochs),
        batch_size=batch_size)

    trained_model = optimizer.optimize()
    logger.info('Successfully trained model')

    return trained_model


def load_model(logger):
    '''
    trained_model = Model.loadModel(
        'hdfs:///user/spot/combalia/autoencoder_model_101_epoch.pkl',
        'hdfs:///user/spot/combalia/autoencoder_model_101_epoch.weights')
    '''
    trained_model = Model.loadModel('hdfs:///user/spotuser/article/model_monday/training/autoencoder_model_monday.pkl',
                                    'hdfs:///user/spotuser/article/model_monday/training/autoencoder_model_monday.weights')

    logger.info('Successfully loaded model')

    return trained_model


def calculate_outliers(trained_model, rdd, sample_rdd, json_settings_dict):

    predictions = trained_model.predict(sample_rdd)

    rdd = rdd.zip(predictions).map(helper)

    df = rdd.toDF()

    # roig
    #df = wannacryLabels(df)

    logger.info('Contamination on train: ' + str(args.contamination))
    logger.info('Percentile on train: ' + str(json_settings_dict['percentile']))

    df_outliers = df.filter(df.score > json_settings_dict['percentile'])

    perc_outliers = float(df_outliers.count()) / float(df.count())

    df_outliers = df_outliers.orderBy('score', ascending=False)
    # ---------------------ROIG-------------------------#

    samples = float(df.count())
    len_positives = float(df_outliers.count())
    len_negatives = samples - len_positives

    logger.info('total number of samples: ' + str(samples))
    logger.info('total detected outliers: ' + str(len_positives))

    if 'label' in df.columns:

        df_outliers = df_outliers.select(df_outliers.tdur,
                                         df_outliers.sport,
                                         df_outliers.dport,
                                         df_outliers.proto,
                                         df_outliers.sip,
                                         df_outliers.dip,
                                         df_outliers.ipkt.cast(IntegerType()),
                                         df_outliers.ibyt.cast(IntegerType()),
                                         df_outliers.opkt.cast(IntegerType()),
                                         df_outliers.obyt.cast(IntegerType()),
                                         df_outliers.score,
                                         df_outliers.label)

        FP = float(df_outliers.where(df_outliers.label.contains('BENIGN')).count())
        TP = abs(len_positives - FP)
        TN = abs(len_negatives - FP)
        FN = samples - TN - FP - TN

        accuracy = (TP + TN)/samples
        precision = TP / (TP + FP)
        recall = TP / TP + FN

        real_outliers = TP + FN

        logger.info('quantity of real outliers / contamination :' + str(real_outliers) + ' / ' + str(real_outliers/samples))
        logger.info('total lenght :' + str(samples))
        logger.info('TP: ' + str(TP))
        logger.info('FP: ' + str(FP))
        logger.info('TN: ' + str(TN))
        logger.info('FN: ' + str(FN))
        logger.info('accuracy: ' + str(accuracy))
        logger.info('precision: ' + str(precision))
        logger.info('recall: ' + str(recall))
        # --------------------------------------------------#
    else:

        df_outliers = df_outliers.select(df_outliers.treceived,
                                         df_outliers.tryear,
                                         df_outliers.trmonth,
                                         df_outliers.trday,
                                         df_outliers.trhour,
                                         df_outliers.trminute,
                                         df_outliers.trsec,
                                         df_outliers.tdur,
                                         df_outliers.sip,
                                         df_outliers.dip,
                                         df_outliers.sport,
                                         df_outliers.dport,
                                         df_outliers.proto,
                                         df_outliers.ipkt.cast(IntegerType()),
                                         df_outliers.ibyt.cast(IntegerType()),
                                         df_outliers.opkt.cast(IntegerType()),
                                         df_outliers.obyt.cast(IntegerType()),
                                         df_outliers.score)
    return df_outliers


def obtain_features(df):
    # Instatiating the transformers
    transform_to_log = ToLog()
    transform_protocol_onehot = ProtocolOneHot()
    transform_flag_onehot = FlagOneHot()
    transform_sport_processing = PortProcessing(source=True)
    transform_dport_processing = PortProcessing(source=False)

    # Transforming the variables
    df = transform_to_log.transform(df)
    df = transform_protocol_onehot.transform(df)
    df = transform_flag_onehot.transform(df)
    df = transform_sport_processing.transform(df)
    df = transform_dport_processing.transform(df)

    return df


def read_standarization_values(hdfs_json_settings):
    json_settings = hdfs_json_settings.split('/')[-1]

    # Get HDFS structures
    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    file_system = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = file_system.get(hadoopConfiguration())

    if (fs.exists(path(hdfs_json_settings)) == True):
        logger.info("Getting the minimum and maximum file from: {0}".format(hdfs_json_settings))
        srcFile = path(hdfs_json_settings)
        dstFile = path(json_settings)
        fs.copyToLocalFile(False, srcFile, dstFile)

        # Get the file from local filesystem
        json_settings_dict = json.loads(open(json_settings).read())

        # Remove the file from filesystem
        os.remove(json_settings)
        os.remove("." + json_settings + ".crc")
    else:
        logger.error("Couldn't find results in {0}".format(hdfs_json_settings))
        sys.exit(0)

    return json_settings_dict


def standarize_features(df, json_settings_dict):
    # Instantiating the standarization transformers
    standarizer = StandardScaler(data_dict=json_settings_dict)
    feature_creator = CreateFeatures()

    df = standarizer.transform(df)

    # Joining all features into one field in the dataframe
    df = feature_creator.transform(df)

    return df


def calculate_percentile(trained_model, rdd, sample_rdd):
    predictions = trained_model.predict(sample_rdd)

    rdd = rdd.zip(predictions).map(helper)

    predictions_df = rdd.toDF()
    percentile = predictions_df.approxQuantile("score", [1 - args.contamination], 0)[0]

    return percentile


def save_settings(json_settings_dict, hdfs_json_settings):
    json_string = json.dumps(json_settings_dict, ensure_ascii=False)

    json_settings = hdfs_json_settings.split('/')[-1]

    with open(json_settings, "w") as f:
        f.write(json_string)

    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    if (fs.exists(path(hdfs_json_settings)) == True):
        logger.warn("It already exists a file with the minimum and maximum values in {0}".format(hdfs_json_settings))
        logger.warn("The file is going to be overrited.")
        try:
            fs.delete(path(hdfs_json_settings), False)
        except:
            logger.error("Couldn't delete the minimum and maximum file")

    try:
        srcMinMaxFile = path(json_settings)
        dstMinMaxFile = path(hdfs_json_settings)
        fs.moveFromLocalFile(srcMinMaxFile, dstMinMaxFile)
        logger.info("Settings exported correctly to " + str(hdfs_json_settings) + ".")
    except:
        logger.error("Couldn't save the minimum and maximum file in the file system.")


def save_results(df_outliers, hdfsScoredDir):
    # Get the path to save the results
    logger.info("Saving results in {0}".format(hdfsScoredDir))

    path = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    fileSystem = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fileUtil = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileUtil
    hadoopConfiguration = sparkSession.sparkContext._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fileSystem.get(hadoopConfiguration())

    # Save the Data Frame into a CSV file
    if (fs.exists(path(hdfsScoredDir)) == False):
        logger.info("Saving results in {0}".format(hdfsScoredDir))
        fs.mkdirs(path(hdfsScoredDir))
        logger.info("The path {0} has been created.".format(hdfsScoredDir))

    hdfsScoredPath = hdfsScoredDir + "/" + "flow_results.csv"
    if (fs.exists(path(hdfsScoredPath)) == True):
        logger.warn("It already exists a results file in {0}".format(hdfsScoredDir))
        logger.warn("The file is going to be overrited.")
        try:
            fs.delete(path(hdfsScoredPath), False)
        except:
            logger.error("Couldn't delete the file")
    try:
        df_outliers.write.csv(hdfsScoredDir + "/anomalyResults", mode="append", sep="\t")
        srcPath = path(hdfsScoredDir + "/anomalyResults")
        dstPath = path(hdfsScoredPath)
        fileUtil.copyMerge(fs, srcPath, fs, dstPath, True, fs.getConf(), "")
        logger.info("The data has been succesfully saved into HDFS filesystem")
    except:
        logger.error("Couldn't merge the files.")

    logger.info("..Testing has finished..")


def main_fun(sparkSession, args, logger):
    json_settings = "json_settings.json"

    df = read_dataframe(sparkSession, args, logger)
    df = obtain_features(df)

    logger.info(str(args))

    hdfs_json_settings = args.network + "/" + json_settings

    # Normalize features
    if args.mode == 'train':
        json_settings_dict = get_stats(df)
    else:
        json_settings_dict = read_standarization_values(hdfs_json_settings)

    df = standarize_features(df, json_settings_dict)
    df.show(5, False)

    rdd = df.rdd
    sample_rdd = rdd.map(transform_sample)

    if args.mode == 'train':
        trained_model = train(sample_rdd, epochs=10, batch_size=1024)
        # variable
        '''
        trained_model.saveModel('hdfs:///user/spot/combalia/autoencoder_model_101_epoch.pkl',
                                'hdfs:///user/spot/combalia/autoencoder_model_101_epoch.weights',
                                True)
        '''
        #----------ROIG---------
        trained_model.saveModel('hdfs:///user/spotuser/article/model_monday/training/autoencoder_model_monday.pkl',
                                'hdfs:///user/spotuser/article/model_monday/training/autoencoder_model_monday.weights',
                                True)
        # ----------ROIG---------
        percentile = calculate_percentile(trained_model, rdd, sample_rdd)
        json_settings_dict['percentile'] = percentile
        save_settings(json_settings_dict, hdfs_json_settings)

        logger.info(str(json_settings_dict))
    else:
        trained_model = load_model(logger)

        df_outliers = calculate_outliers(trained_model, rdd, sample_rdd, json_settings_dict)
        df_outliers.show()

        save_results(df_outliers, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, help='Train or test')
    parser.add_argument('--input', required=True, help='Path for the train input csv')
    parser.add_argument('--output', required=False, help='Path for the test input csv')
    parser.add_argument('--network', required=False, help='Path of the trained network')

    parser.add_argument('--contamination', required=True, type=float)
    parser.add_argument('--number_outliers', required=True, type=int)
    parser.add_argument('--input_type', required=False)

    args = parser.parse_args()

    sparkSession = SparkSession\
                   .builder\
                   .appName("Security ML")\
                   .getOrCreate()

    Level = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.Level
    LogManager = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.LogManager
    Logger = sparkSession.sparkContext._gateway.jvm.org.apache.log4j.Logger

    logger = LogManager.getLogger("DeepLearningSecurity")
    logger.setLevel(Level.INFO)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    logger.info(str(args))

    redire_spark_logs()
    show_bigdl_info_logs()

    init_engine()

    logger.info("Successfully loaded BIGDL")

    main_fun(sparkSession, args, logger)

    sparkSession.stop()
