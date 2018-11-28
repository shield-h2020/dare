#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:37:14 2018

@author: Dimitris Papadopoulos

"""

import logging
from datetime import datetime

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import lit

sc = SparkContext('local[*]')
spark = SparkSession(sc)

from argparse import ArgumentParser     
from os.path import basename

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline, PipelineModel, Model

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



def train(filename, logger, trees):
       
    print('\x1b[1;32m'+'\n'+'Starting preprocessing stage...'+'\x1b[0m')    
    
    print("Importing labeled dataset")

    df_data1 = spark.read\
      .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
      .option('header', 'true')\
      .option('inferSchema', 'true')\
      .load(filename)
    
   
    # Drop null values
    df_data=df_data1.filter(df_data1.Class.isNotNull())
    print("Dropped rows with null values...")
    
    
    #Casting appropriate datatypes to data.
    df_data = df_data.select(
               df_data.tr,
               df_data.tryear,
               df_data.trmonth,
               df_data.trday,
               df_data.trhour,
               df_data.trmin,
               df_data.trsec,
               df_data.td.cast(IntegerType()),
               df_data.sa,
               df_data.da,
               df_data.sp.cast(IntegerType()), 
               df_data.dp.cast(IntegerType()),
               df_data.pr,
               df_data.ipkt.cast(IntegerType()),
               df_data.ibyt.cast(IntegerType()),
               df_data.opkt,
               df_data.obyt,
               df_data.Class)  
    
    
    # These are ignored columns: ["tr","tryear","trmonth", "trday", "trhour","trmin","trsec","sa","da","opkt","obyt"]
    
    df_data.printSchema()

    print("Creating training and testing sets...")
    #Split data into: train, test and predict datasets.
    splitted_data = df_data.randomSplit([0.90, 0.10, 0.00],123)
    train_data = splitted_data[0]
    test_data = splitted_data[1]
    predict_data = splitted_data[2]
    #    train_data=df_data
    #    test_data=df_data
    #    predict_data=df_data

    
    print("Encoding categorical variables to numerical...")
    #Convert all the string fields to numeric ones by using the StringIndexer transformer.
    stringIndexer_label = StringIndexer(inputCol="Class", outputCol="label").setHandleInvalid("skip").fit(df_data)
    stringIndexer_prot = StringIndexer(inputCol="pr", outputCol="prot").setHandleInvalid("skip")

    #Create a feature vector by combining all features together.
    vectorAssembler_features = VectorAssembler(inputCols=["sp","dp","prot", "ipkt", "ibyt"], outputCol="features")

  
    print('\x1b[1;32m'+'Starting ML stage...'+'\x1b[0m')

    #Define estimators for classification. Random Forest is used.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",maxBins=300, numTrees=trees)
    
    
    
    #Indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=stringIndexer_label.labels)

    #Build a pipeline that consists of transformers and an estimator.
    pipeline_rf = Pipeline(stages=[stringIndexer_label, stringIndexer_prot, vectorAssembler_features, rf, labelConverter])
    

#    train_data.printSchema()
    train_data.na.drop(how="any")
    
    #Cleaning data from null values (after split)
    train_data = train_data.filter(train_data.sp. isNotNull())
    train_data = train_data.filter(train_data.dp. isNotNull())
    train_data = train_data.filter(train_data.ibyt. isNotNull())
    train_data = train_data.filter(train_data.ipkt. isNotNull())
    train_data = train_data.filter(train_data.Class. isNotNull())
    
    
    #Crossvalidation setup
    numFolds=10
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth,[6,8,10]).addGrid(rf.featureSubsetStrategy, ["auto","log2"]).build()
    evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    crossval = CrossValidator(
    estimator=pipeline_rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluatorRF,
    numFolds=numFolds)
    print("Fitting Random Forest model of",trees ,"trees to training set...")

    #Training Random Forest model by using the previously defined pipeline and train data.
    model_rf = crossval.fit(train_data)
    
    print("Done!")
    
    print("Checking model's accuracy using testing set...")

    #Check model's accuracy using test data.
    predictions = model_rf.transform(test_data)
    evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluatorRF.evaluate(predictions)

    print("Accuracy = %g" % accuracy)
    print("Test Error = %g" % (1.0 - accuracy))
  
  
    # Save best pipeline model
    
    final_model = model_rf.bestModel

    print("Saving trained RF model")
    #Specify desired HDFS path to save trained model
    savepath = "/user/spotuser/rf/"
    modelpath = savepath + "trainedRF_model_spotformat"
    final_model.write().overwrite().save(modelpath)
    
    print("Trained RF model saved as: "+modelpath+'\n')
    
    
    

def predict():   
      
    try:
        try: 
            env  = parse_args()
            filename=env.filename
            print("\nReading flows for classification...")

            df_data = spark.read.format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
            .option('header', 'false')\
            .option('inferSchema', 'true')\
            .options(delimiter='\t')\
            .load(filename)  
        except KeyError as err:
            raise ValueError('Missing environment variable {0}'.format(err))
        
        # Assign columns names
        df_data = df_data.withColumnRenamed("_c0", "trtime")
        df_data = df_data.withColumnRenamed("_c1", "tryear")    
        df_data = df_data.withColumnRenamed("_c2", "trmonth")
        df_data = df_data.withColumnRenamed("_c3", "trday")    
        df_data = df_data.withColumnRenamed("_c4", "trhour")    
        df_data = df_data.withColumnRenamed("_c5", "trmin")    
        df_data = df_data.withColumnRenamed("_c6", "trsec")    
        df_data = df_data.withColumnRenamed("_c7", "td")    
        df_data = df_data.withColumnRenamed("_c8", "sa")
        df_data = df_data.withColumnRenamed("_c9", "da")    
        df_data = df_data.withColumnRenamed("_c10", "sp")    
        df_data = df_data.withColumnRenamed("_c11", "dp")    
        df_data = df_data.withColumnRenamed("_c12", "pr")    
        df_data = df_data.withColumnRenamed("_c13", "ipkt") 
        df_data = df_data.withColumnRenamed("_c14", "ibyt")    
        df_data = df_data.withColumnRenamed("_c15", "opkt")    
        df_data = df_data.withColumnRenamed("_c16", "obyt")    
        df_data = df_data.withColumnRenamed("_c17", "prob")    
        
        
        
        
        #Casting appropriate datatypes to data.
        df_data = df_data.select(
               df_data.trtime.cast(StringType()),
               df_data.tryear,
               df_data.trmonth,
               df_data.trday,
               df_data.trhour,
               df_data.trmin,
               df_data.trsec,
               df_data.td.cast(IntegerType()),
               df_data.sa,
               df_data.da,
               df_data.sp.cast(IntegerType()), 
               df_data.dp.cast(IntegerType()),
               df_data.pr,
               df_data.ipkt.cast(IntegerType()),
               df_data.ibyt.cast(IntegerType()),
               df_data.opkt,
               df_data.obyt,
               df_data.prob.cast(DoubleType())
               )   
        
        df_data.printSchema()

        print("Encoding categorical variables to numerical...")
        #Convert all the string fields to numeric ones by using the StringIndexer transformer.
#        stringIndexer_label = StringIndexer(inputCol="Class", outputCol="label").setHandleInvalid("skip").fit(df_data)
        stringIndexer_prot = StringIndexer(inputCol="pr", outputCol="prot").setHandleInvalid("skip")
    
        #Create a feature vector by combining all features together.
        vectorAssembler_features = VectorAssembler(inputCols=["sp","dp","prot", 
                                                              "ipkt", "ibyt"], outputCol="features")
        
        # Drop null values        
        print("Dropping rows containing null values...")
        test_data = df_data
        test_data = test_data.filter(test_data.sa. isNotNull())
        test_data = test_data.filter(test_data.da. isNotNull())
        test_data = test_data.filter(test_data.td. isNotNull())
        test_data = test_data.filter(test_data.sp. isNotNull())
        test_data = test_data.filter(test_data.dp. isNotNull())
        test_data=  test_data.filter(test_data.pr. isNotNull())
        test_data = test_data.filter(test_data.ipkt. isNotNull())
        test_data = test_data.filter(test_data.ibyt. isNotNull())
  
        # Load saved model to evaluate-predict
        print("Loading the trained Random Forest model \n")

        #Specify HDFS path of trained model
        savepath = "/user/spotuser/rf/"
        modelpath= savepath + "trainedRF_model_spotformat"
        sameModel = PipelineModel.load(modelpath)
        predictions = sameModel.transform(test_data)

        print('\x1b[1;32m'+"Threat classification complete! Saving results file..."+'\x1b[0m')
        
        
        predictions=predictions.withColumn("csv_id", lit(str(datetime.now().strftime("%Y%m%d"))))\
        .withColumn("severity", lit("High"))


        predictions= predictions.select("csv_id","severity","predictedLabel","trtime","tryear","trmonth","trday","trhour","trmin","trsec","td", "sa", "da", "sp", "dp", "pr", "ipkt", "ibyt","opkt","obyt","prob")

        file=basename(filename)
        resultsfilename="results_"+file
        predictions.write.mode('overwrite').format("com.databricks.spark.csv").option("header", "true").options(delimiter='\t').save(savepath+resultsfilename)

        print("Succesfully created aggregated results folder:", resultsfilename)
           
        threatresults=predictions.filter(predictions.predictedLabel!="background")
        
        threatresults= threatresults.select("csv_id","severity","predictedLabel","trtime","tryear","trmonth","trday","trhour","trmin","trsec","td", "sa", "da", "sp", "dp", "pr", "ipkt", "ibyt","opkt","obyt","prob").dropDuplicates(subset=['sa','pr'])
        threatresultsfilename="threatresults_"+file
        threatresults.write.mode('overwrite').format("com.databricks.spark.csv").option("header", "false").options(delimiter='\t').save(savepath+threatresultsfilename)

        print("Succesfully created aggregated threatresults folder:", threatresultsfilename)


  
    except KeyboardInterrupt:
        print(' Process terminated by user.')
        sys.exit(0)    



def parse_args():
    
    '''
        Convert argument strings to objects and assign them as attributes of the
    namespace.

    :return: The populated namespace.
    :rtype : argparse.Namespace
    '''
    
    parser = ArgumentParser(description="Random Forest Classifier for network traffic.",
    epilog='END')
    
    parser.add_argument('-p', '--phase', required=True, metavar='',
        dest='phase', help='Type of operation (train | predict) (required)')
    parser.add_argument('-c', '--csv', required=True, metavar='',
        dest='filename', help='Filepath of the .csv file required for training (required)')
    parser.add_argument('-t', '--trees', default=10, metavar='',
        dest='trees', help=' integer number of trees that will form the Random Forest (default=10)')

    return parser.parse_args()

if __name__ == "__main__":
    
    logger = logging.getLogger()

    env = parse_args()
    phase = env.phase
    
    if phase == "train":
        trees = env.trees
        trees = int(trees)
        filename = env.filename
        train(filename,logger,trees)  
    elif phase == "predict":
        predict()
 
