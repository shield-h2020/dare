#MIT License
#
#Copyright (c) [2018] [Marc Comablia, Marc Roig, Bernat Gastón]
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from bigdl.util.common import *

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql.types import IntegerType, ArrayType, FloatType
from pyspark.sql import functions as F

import numpy as np
from pyspark.sql import Row


# Transforms the dataframe to the required format to train the model

def transform_sample(row):
    values = np.array(row['features'])
    sample = Sample.from_ndarray(values, values)

    return sample

# Calculates the percentile of the MSE to consider anomalies

def percentile_threshold(ardd, percentile):
    assert 100 >= percentile > 0, "percentile should be larger then 0 and smaller or equal to 100"

    return ardd.sortBy(lambda x: x).zipWithIndex().map(lambda x: (x[1], x[0])) \
            .lookup(np.ceil(ardd.count() / 100 * percentile - 1))[0]


# Obtains the mean and std values to normalize the variables described in feat_to_norm

feat_to_norm = ["tdur_log", "ipkt_log", "ibyt_log"]

def get_stats(df):
    # Obtain the mean and std values from the train database and use them to normalize train and test
    normalization_values = {}
    for feat in feat_to_norm:
        normalization_values[feat] = {}
        mean = df.select(F.avg(feat)).collect()[0][0]
        std = df.select(F.stddev(feat)).collect()[0][0]

        normalization_values[feat]['mean']  = mean
        normalization_values[feat]['std'] = std

    return normalization_values


# Applies a logarithm transform to the variables tdur, ipkt and ibyt

class ToLog(Transformer):

    @keyword_only
    def __init__(self):
      super(ToLog, self).__init__()

    def _transform(self, df):
        df = df.withColumn('tdur_log', F.log(df.tdur + 1)).withColumn('ipkt_log', F.log(df.ipkt + 1)).withColumn(
            'ibyt_log',F.log(df.ibyt))
        return df

# Transforms the protocol string into one-hot encoding using a dictionary

class ProtocolOneHot(Transformer):

    @keyword_only
    def __init__(self):
        super(ProtocolOneHot, self).__init__()

    def _transform(self, df):
        def transform_protocol(proto):
            list_prots = ['UDP', 'TCP', 'ICMP', 'IGMP']

            output_list = [0, 0, 0, 0, 0]
            found_prot = False

            for i in range(len(list_prots)):
                if list_prots[i] in proto:
                    output_list[i] += 1
                    found_prot = True
                    break

            if found_prot == False:
                output_list[len(list_prots)] += 1

            return output_list

        proto_transform = F.udf(lambda z: transform_protocol(z), ArrayType(IntegerType()))
        df = df.withColumn('proto_onehot', proto_transform(df.proto))

        return df

# Transforms the flag string into one-hot encoding using a dictionary

class FlagOneHot(Transformer):

    @keyword_only
    def __init__(self):
        super(FlagOneHot, self).__init__()

    def _transform(self, df):
        def process_flag(flag):
            list_flags = ['A', 'S', 'F', 'R', 'P', 'U', 'X']

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

        flag_transform = F.udf(lambda z: process_flag(z),  ArrayType(IntegerType()))
        df = df.withColumn('flag_onehot', flag_transform(df.flag))

        return df


# Trasforms ports into one-hot encoding (0 < port < 1024), other rules can be defined if need be


class PortProcessing(Transformer):

    @keyword_only
    def __init__(self, source = False):
        super(PortProcessing, self).__init__()
        self.source = source

    def _transform(self, df):
        def process_port(port):
            if port > 1024:
                return 1
            else:
                return 0

        port_transform = F.udf(lambda z: process_port(z),  IntegerType())
        if self.source:
            df = df.withColumn('sport_transform', port_transform(df.sport))
        else:
            df = df.withColumn('dport_transform', port_transform(df.dport))

        return df


# Scales the variables using the mean and std values provided by a dictionary

class StandardScaler(Transformer):
    @keyword_only
    def __init__(self, data_dict):
        super(StandardScaler, self).__init__()
        self.tdur_mean = data_dict['tdur_log']['mean']
        self.tdur_std = data_dict['tdur_log']['std']
        self.ipkt_mean = data_dict['ipkt_log']['mean']
        self.ipkt_std = data_dict['ipkt_log']['std']
        self.ibyt_mean = data_dict['ibyt_log']['mean']
        self.ibyt_std = data_dict['ibyt_log']['std']

    def _transform(self, df):
        df = df.withColumn('tdur_log_standarized', (df.tdur_log - self.tdur_mean)/self.tdur_std)\
            .withColumn('ipkt_log_standarized', (df.ipkt_log - self.ipkt_mean)/self.ipkt_std)\
            .withColumn('ibyt_log_standarized', (df.ipkt_log - self.ibyt_mean)/self.ibyt_std)\

        return df


# Joins all the features into a single variable in the dataframe, required to make work bigdl

class CreateFeatures(Transformer):
    @keyword_only
    def __init__(self):
        super(CreateFeatures, self).__init__()

    def _transform(self, df):
        def process(proto_onehot, flag_onehot, tdur, ipkt, ibyt, sport, dport):
            features = proto_onehot + flag_onehot

            features.append(tdur)
            features.append(ipkt)
            features.append(ibyt)
            features.append(sport)
            features.append(dport)

            features = [float(i) for i in features]

            return features

        transformer = F.udf(process, ArrayType(FloatType()))
        df = df.withColumn('features', transformer(df.proto_onehot, df.flag_onehot, df.tdur_log_standarized, df.ipkt_log_standarized, df.ibyt_log_standarized, df.sport_transform, df.dport_transform))

        return df

def helper(item):
    row = item[0]

    temp = row.asDict()
    temp['predictions'] = item[1].tolist()

    features = np.array(temp['features'])
    predictions = np.array(temp['predictions'])

    mse = np.linalg.norm(features - predictions)
    temp['score'] = float(mse)

    return Row(**temp)

# ---------------------ROIG-------------------------#
def wannacryLabels(df):
    if not 'label' in df.columns:
        df = df.withColumn('label', F.when((df.sip == "192.168.116.138") |
                                           (df.sip == "192.168.116.143") |
                                           (df.dip == "192.168.116.149") |
                                           (df.dip == "192.168.116.150") |
                                           (df.dip == "192.168.116.172") |
                                           (df.dip == "192.168.116.254") |
                                           (df.dip == "192.168.116.255") |
                                           (df.sip == "192.168.116.138") |
                                           (df.sip == "192.168.116.143") |
                                           (df.sip == "192.168.116.149") |
                                           (df.sip == "192.168.116.150") |
                                           (df.sip == "192.168.116.172") |
                                           (df.sip == "192.168.116.254") |
                                           (df.sip == "192.168.116.255"),
                                           "WANNACRY").otherwise("BENIGN"))
        directori = "hdfs:///user/spotuser/article/test.csv"
        print(df.show(5))
        df = df.select(df.tdur,
                       df.sport,
                       df.dport,
                       df.proto,
                       df.sip,
                       df.dip,
                       df.ipkt.cast(IntegerType()),
                       df.ibyt.cast(IntegerType()),
                       df.opkt.cast(IntegerType()),
                       df.obyt.cast(IntegerType()),
                       df.score,
                       df.label)
        df.write.csv(directori, mode="append")
