from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pandas as pd
import tensorflow as tf
from . import model as model_constructor
# import model as model_constructor
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import time
TRAIN_FILE = './dataset/train.csv'
OUTPUT_DIR = './outputs/'
# TEST_FILE  = 'mnist digit dataset/digit-recognizer/test.csv'


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.001,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--val-size',
        default=0.33,
        type=float,
        help='how much fraction of the dataset should be used for validation'
    )
    parser.add_argument(
        '--storage-prefix',
        default='',
        type=str,
        help='prefix to be added before storing results'
    )
    args, _ = parser.parse_known_args()
    return args

def prepare_inputs(args):
    inputFile = os.path.join(args.job_dir, TRAIN_FILE);
    df = pd.read_csv(tf.io.gfile.GFile(inputFile));
    print(df.describe())
    xpd = df.iloc[:,1:]
    ypd = df['label']
    x = xpd.to_numpy()
    y = ypd.to_numpy()
    x = x / 255.0
    x = x.reshape((-1, 28, 28,1))
    return train_test_split(x, y, test_size=args.val_size)

def train_and_evaluate(args):
    randString = args.storage_prefix + str(int(time.time()))
    train_x, test_x, train_y, test_y = prepare_inputs(args)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1)
    model = model_constructor.create_model('Adam', SparseCategoricalCrossentropy())
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate * (0.95 ** ((epoch//3)*3)),
        verbose=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, '{0}{1}/keras_tensorboard'.format(OUTPUT_DIR,randString)),
        histogram_freq=1)
    earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)
    model.fit(datagen.flow(train_x,train_y,batch_size=args.batch_size),
              epochs=args.num_epochs,
              validation_data=(test_x,test_y),
              callbacks=[lr_decay_cb, tensorboard_cb, earlystopping_cb]
             )
    export_path = os.path.join(args.job_dir, '{0}{1}/output_model'.format(OUTPUT_DIR,randString))
    model.save(export_path)
    print('Model exported to: {}'.format(export_path))
    

if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)