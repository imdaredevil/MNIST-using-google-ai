import tensorflow as tf
import pandas as pd
import numpy as np
import json

def prepare_test_data(inputFile, limit = 10):
    df = pd.read_csv(tf.io.gfile.GFile(inputFile))
    print(df.describe())
    x = df.to_numpy()
    x = x / 255.0
    x = x.reshape((-1, 28, 28,1))
    return x

def prepare_train_data(inputFile):
    df = pd.read_csv(tf.io.gfile.GFile(inputFile))
    print(df.describe())
    xpd = df.iloc[:,1:]
    ypd = df['label']
    x = xpd.to_numpy()
    y = ypd.to_numpy()
    x = x / 255.0
    x = x.reshape((-1, 28, 28,1))
    return x,y

def convert_to_list(x, limit=None):
    xlist = x.tolist()
    if limit is not None:
        xlist = xlist[0:limit]
    return xlist

def write_json_for_submission(jsonFilePath,xlist):
    jsonFile = open(jsonFilePath, 'w')
    for x in xlist:
        jsonx = json.dumps(x)
        jsonFile.write(jsonx)
        jsonFile.write('\n')
    jsonFile.close()

