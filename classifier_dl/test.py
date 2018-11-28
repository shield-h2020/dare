
# Deep Learning Studio - GUI platform for designing Deep Learning AI without programming
#
# Copyright (C) 2018 Deep Cognition Inc.
#
# All rights reserved.

from __future__ import print_function
import os

import sys
import time
import yaml
import pickle
import traceback
import pandas as pd
import numpy as np
import scipy.misc
import scipy.io.wavfile
from sklearn.preprocessing import LabelEncoder
from importlib.machinery import SourceFileLoader
from multiprocessing import cpu_count

os.environ["MKL_NUM_THREADS"] = str(cpu_count())

def doResize(options):
    resize = None
    if options and 'Resize' in options and options['Resize'] == True:
        resize = (int(options['Width']), int(options['Height']))
    return resize


def col_pre_process(data, options):
    if len(options.keys()) == 0:
        return data
    else:
        if "pretrained" in options and options["pretrained"] != 'None':
            from keras.applications import inception_v3, vgg16, vgg19, resnet50
            if options["pretrained"] == 'InceptionV3':
                data = inception_v3.preprocess_input(data)
            elif options["pretrained"] == 'ResNet50':
                data = resnet50.preprocess_input(data)
            elif options["pretrained"] == 'VGG16':
                data = vgg16.preprocess_input(data)
            elif options["pretrained"] == 'VGG19':
                data = vgg19.preprocess_input(data)

        if "Scaling" in options and float(options["Scaling"]) != 0 and float(options["Scaling"]) != 1:
            data = data / float(options["Scaling"])

        if 'Normalization' in options and options['Normalization'] == True:
            mean = np.mean(data)
            std = np.std(data)
            data = data - mean
            data = data / std
            return data
        return data


def process_test_input(base_dir, test_raw, data_mapping):
    test_data = []
    le = None

    from keras import backend as K
    if K.backend() == 'theano' or K.backend() == 'mxnet':
        K.set_image_dim_ordering('th')
    else:
        K.set_image_dim_ordering('tf')

    # determine the shape of the data to feed into the network
    for i in range(len(data_mapping['inputs'])):
        inp_port = data_mapping['inputs']['InputPort' + str(i)]
        if inp_port['details'][0]['type'] == 'Image':
            col_name = inp_port['details'][0]['name']
            if 'options' in inp_port['details'][0]:
                options = inp_port['details'][0]['options']
            else:
                options = {}

            resize = doResize(options)
            img = scipy.misc.imread(base_dir + "/" + test_raw[col_name][0])
            input_shape = img.shape

            num_channels = 1
            if resize:
                width, height = resize
                if len(input_shape) == 3:
                    num_channels = 3
            else:
                if len(input_shape) == 2:
                    width, height = input_shape
                else:
                    width, height, num_channels = input_shape

            test_data.append(np.ndarray((len(test_raw),) +
                                        (num_channels, width, height), dtype=np.float32))

            for j, filename in enumerate(test_raw[col_name]):
                img = scipy.misc.imread(base_dir + "/" + filename)
                if resize:
                    img = scipy.misc.imresize(img, resize)
                if num_channels != 1:
                    img = np.transpose(img, (2, 0, 1))
                test_data[i][j] = img

            if K.image_dim_ordering() == 'tf':
                test_data[i] = np.transpose(test_data[i], (0, 2, 3, 1))

            test_data[i] = col_pre_process(test_data[i], options)

        elif inp_port['details'][0]['type'] == 'Audio':
            if 'options' in inp_port['details'][0]:
                options = inp_port['details'][0]['options']
            else:
                options = {}

            (rate, data) = scipy.io.wavfile.read(
                base_dir + "/" + test_raw[col_name][0])
            test_data.append(np.ndarray((len(test_raw),) +
                                        data.shape, dtype=data.dtype))

            for j, filename in enumerate(test_raw[col_name]):
                (rate, data) = scipy.io.wavfile.read(base_dir + "/" + filename)
                test_data[i][j] = data
            test_data[i] = col_pre_process(test_data[i], options)

        elif inp_port['details'][0]['type'] == 'Numpy':
            if 'options' in inp_port['details'][0]:
                options = inp_port['details'][0]['options']
            else:
                options = {}

            col_name = inp_port['details'][0]['name']
            npzFile = np.load(base_dir + "/" + test_raw[col_name][0])
            x = npzFile[npzFile.files[0]]
            input_shape = x.shape

            test_data.append(np.ndarray(
                (len(test_raw),) + x.shape, dtype=np.float32))
            for j, filename in enumerate(test_raw[col_name]):
                npzFile = np.load(base_dir + "/" + filename)
                x = npzFile[npzFile.files[0]]
                test_data[i][j] = x
            test_data[i] = col_pre_process(test_data[i], options)

        else:
            col_idx = 0
            test_data.append(np.ndarray(
                (len(test_raw), inp_port['size']), dtype=np.float32))
            for col in range(len(inp_port['details'])):
                if 'options' in inp_port['details'][col]:
                    options = inp_port['details'][col]['options']
                else:
                    options = {}

                col_name = inp_port['details'][col]['name']

                if inp_port['details'][col]['type'] == 'Categorical':
                    data_col = test_raw[col_name]
                    num_categories = len(
                        inp_port['details'][col]['categories'])

                    le_temp = LabelEncoder()
                    le_temp.fit(inp_port['details'][col]['categories'])
                    data_col = le_temp.transform(data_col)

                    one_hot_array = np.zeros(
                        (len(data_col), num_categories), dtype=np.float32)
                    one_hot_array[np.arange(len(data_col)), data_col] = 1

                    test_data[i][:, col_idx:col_idx +
                                 num_categories] = col_pre_process(one_hot_array, options)
                    col_idx += num_categories

                elif inp_port['details'][col]['type'] == 'Array':
                    array = np.array(test_raw[col_name].str.split(
                        ';').tolist(), dtype=np.float32)
                    test_data[i][:, col_idx:col_idx + array.shape[1]
                                 ] = col_pre_process(array, options)
                    col_idx += array.shape[1]

                else:
                    data = test_raw[col_name].reshape((len(test_raw), 1))
                    test_data[i][:, col_idx:col_idx +
                                 1] = col_pre_process(data, options)
                    col_idx += 1

    # assuming single output, generate labelEncoder
    out_port = data_mapping['outputs']['OutputPort0']
    if out_port['details'][0]['type'] == 'Categorical':
        le = LabelEncoder()
        le.fit(out_port['details'][0]['categories'])

    return test_data, le


def customPredict(test_data, config, modelFile):
    res = None
    loss_func = config['params']['loss_func']
    if 'is_custom_loss' in config['params']:
        isCustomLoss = config['params']['is_custom_loss']
    else:
        isCustomLoss = False

    if isCustomLoss:
        customLoss = SourceFileLoader(
            "customLoss", 'customLoss.py').load_module()
        loss_function = eval('customLoss.' + loss_func)
        mod = load_model(modelFile, custom_objects={loss_func: loss_function})
    else:
        mod = load_model(modelFile)

#    if os.environ.get("GPU_ENABLED", "0") == "1":
#        mod.compile (loss='categorical_crossentropy', optimizer='adam', context=["GPU(0)"])
        
    return mod.predict(test_data)


def test_model(input_file):

    try:
        if os.path.exists('model.h5') and os.path.exists('mapping.pkl'):

            with open('mapping.pkl', 'rb') as f:
                data_mapping = pickle.load(f)

            test_raw = pd.read_csv(input_file)

            test_data, le = process_test_input(
                os.path.dirname(input_file), test_raw, data_mapping)

            currentDir = os.getcwd()

            with open('config.yaml', 'r') as f:
                config = yaml.load(f)
                models = []
                if "kfold" in config["data"] and config["data"]["kfold"] > 1:
                    kfold = config["data"]["kfold"]

                    if os.path.exists('model.h5'):
                        models.append(currentDir + '/model.h5')
                    else:
                        for sub_run in range(1, kfold + 1):
                            sub_dir = currentDir + str(sub_run)
                            if os.path.exists(sub_dir + "/model.h5"):
                                models.append(sub_dir + "/model.h5")
                else:
                    models.append(currentDir + '/model.h5')

            result = np.array([])
            for modelFile in models:
                res = customPredict(test_data, config, modelFile)
                if result.size != 0:
                    result = res + result
                else:
                    result = res

            res = result / len(models)

            out_type = data_mapping['outputs']['OutputPort0']['details'][0]['type']

            num_samples = len(test_raw)
            if num_samples != 0:
                out_dir = "./"
                if out_type == "Numpy":
                    if not os.path.exists(out_dir + "output/"):
                        os.makedirs(out_dir + "output/")
                    temp = np.ndarray((res.shape[0],), dtype=np.object_)
                    for i in range(res.shape[0]):
                        filename = "./output/" + str(i) + ".npy"
                        np.save(out_dir + filename, res[i])
                        temp[i] = filename

                    test_raw['predictions'] = temp
                elif out_type == 'Array':
                    temp = np.ndarray((res.shape[0],), dtype=np.object_)
                    res = np.round(res,  decimals=2)
                    for i in range(res.shape[0]):
                        temp[i] = np.array2string(
                            res[i], precision=2, separator=';')
                    test_raw['predictions'] = temp
                elif out_type == 'Categorical' and le != None:
                    res_prob = np.round(
                        np.max(res, axis=1).astype(float), decimals=4)
                    res_id = np.argmax(res, axis=1)
                    res1 = le.inverse_transform(res_id.tolist())
                    test_raw['predictions'] = res1
                    test_raw['probabilities'] = res_prob
                elif out_type == "Image":
                    if not os.path.exists(out_dir + "output/"):
                        os.makedirs(out_dir + "output/")
                    temp = np.ndarray((res.shape[0],), dtype=np.object_)
                    from keras import backend as K
                    if K.image_dim_ordering() == 'th':
                        res = np.transpose(res, (0, 2, 3, 1))
                        
                    for i in range(res.shape[0]):
                        filename = "./output/" + str(i) + str(round(time.time())) + ".png"
                        if res.shape[-1] == 1:
                            img = np.reshape(
                                res[i], (res.shape[1], res.shape[2]))
                        else:
                            img = res[i]
                        scipy.misc.imsave(out_dir + filename, img)
                        temp[i] = filename

                    test_raw['predictions'] = temp
                else:
                    test_raw['predictions'] = res

                test_raw.to_csv('test_result.csv', index=False)

        else:
            print('model or data mapping does not exist... try downloading again!')

    except Exception as e:
        print("aborting due to exception... Please check input file format!")
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: \"python3 test.py test.csv\"")
    else:

        # imports that depend on backend
        from keras.models import load_model

        print("outputs will be stored at \'./output/\'\n")
        test_model(sys.argv[1])
