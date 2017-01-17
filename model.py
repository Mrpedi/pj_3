import numpy as np
import sys, os
import pandas as pd
import argparse
import pickle
from sklearn.cross_validation import train_test_split
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras import optimizers
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread




"""
This script uses Keras API (https://keras.io), to train a deep neural network to predict steering angles from image
dataset created by Udacity's Simulation environment. The deep neural net is based on NVIDIA's model, see link below:
https://arxiv.org/pdf/1604.07316v1.pdf, which was capable of correctly predict steering angles.
"""

def read_csv():
    # Reading Training Data
    # Training data is composed by features Left/Center/Right Images in 'dataset/IMG'
    # Labels are associated to each image feature being 'steering angle', 'Throttle' and 'Breaking'
    # For this projetct the goal is classify the steering angles, so only steering angles will be
    # classified for the Conv. Net.

    df_input_data = pd.read_csv(filepath_or_buffer=('data/driving_log.csv'),
                             names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])

    X_train = df_input_data.center.values
    y_train = df_input_data.steering.values.astype(np.float32)

    return train_test_split(X_train, y_train)

def load_images(X_data):

    X_out = np.ones(shape=(len(X_data), 160, 320, 3),dtype=np.int8)

    for i,v in enumerate(X_data):
        file_path = v.split('/')[-1]
        im_data = imread(os.path.join('data/IMG',file_path))
        X_out[i][:] = im_data

    return X_out


def export_model(model):
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')

def image_generator():
    # In order to handle large datasets which may not fit in memory, this example loads only a set of images
    # according to user requirements

    return 0

def parse():

    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument("-b", "--batch_size", default=64, help='Batch Size')
    parser.add_argument("-e", "--epoch_size", default=50, help="Number of epochs")
    args = parser.parse_args()
    return args


def main():

    args = parse()

    # Load or create pickle from training data to speed up script
    if os.path.exists('train.p') and os.path.exists('valid.p'):
        train_p = open('train.p', mode='rb')
        valid_p = open('valid.p', mode='rb')
        X_train, y_train = pickle.load(train_p)
        X_valid, y_valid = pickle.load(valid_p)
    else:
        # Read csv info
        print("---- Reading CSV Info ----")
        X_train, X_valid, y_train, y_valid = read_csv()

        X_train = load_images(X_train)
        pickle.dump([X_train, y_train], open('train.p', mode='wb'))

        X_valid = load_images(X_valid)
        pickle.dump([X_valid, y_valid], open('valid.p', mode='wb'))


    print("Train/Validation Size: %d/%d" % (len(X_train),len(X_valid)))

    n_classes = len(np.unique(y_train))

    # One Hot Encoding

    y_train = to_categorical(y_train, n_classes)

    # ----- Keras Image Preprocessing -----



    # ----- Model definition -----

    model_dict = {'in_shape':(66, 200, 3),  # For tensorflow backend use shape as (row, cols, chan)
                  'nb_filter_1':24, 'nb_row_1':5, 'nb_col_1':5, 'stride_1':(2,2), 'z_1':'tanh','pad_1':'valid',
                  'nb_filter_2':36, 'nb_row_2':5, 'nb_col_2':5, 'stride_2':(2,2), 'z_2':'tanh','pad_2':'valid',
                  'nb_filter_3': 48, 'nb_row_3': 5, 'nb_col_3': 5, 'stride_3':(2,2), 'z_3':'tanh','pad_3':'valid',
                  'nb_filter_4': 64, 'nb_row_4': 3, 'nb_col_4': 3, 'stride_4':(1,1), 'z_4':'tanh','pad_4':'valid',
                  'nb_filter_5': 64, 'nb_row_5': 3, 'nb_col_5': 3, 'stride_5':(1,1), 'z_5':'tanh','pad_5':'valid',
                  'drop_p': 0.5}

    model = Sequential()
    # First Layer - 3@66x200 -> 24@31x98
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_1'],
                     nb_row=model_dict['nb_row_1'],
                     nb_col=model_dict['nb_col_1'],
                     activation=model_dict['z_1'],
                     border_mode=model_dict['pad_1'],
                     subsample=model_dict['stride_1'],
                     init='glorot_normal',
                     input_shape=model_dict['in_shape']))

    model.add(Dropout(model_dict['drop_p']))

    # Second Layer - 24@31x98 -> 36@14x47
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_2'],
                     nb_row=model_dict['nb_row_2'],
                     nb_col=model_dict['nb_col_2'],
                     activation=model_dict['z_2'],
                     border_mode=model_dict['pad_2'],
                     subsample=model_dict['stride_2']))

    # Third Layer - 36@14x47 -> 48@5x22
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_3'],
                     nb_row=model_dict['nb_row_3'],
                     nb_col=model_dict['nb_col_3'],
                     activation=model_dict['z_3'],
                     border_mode=model_dict['pad_3'],
                     subsample=model_dict['stride_3']))
    
    # Fourth Layer  - 48@5x22 -> 64@3x20
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_4'],
                     nb_row=model_dict['nb_row_4'],
                     nb_col=model_dict['nb_col_4'],
                     activation=model_dict['z_4'],
                     border_mode=model_dict['pad_4'],
                     subsample=model_dict['stride_4']))
    
    # Fifth Layer
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_5'],
                     nb_row=model_dict['nb_row_5'],
                     nb_col=model_dict['nb_col_5'],
                     activation=model_dict['z_5'],
                     border_mode=model_dict['pad_5'],
                     subsample=model_dict['stride_5']))

    # Flat model - prepare for Dense Layers
    model.add(Flatten())

    # Sixth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(1164,activation='tanh'))
    # Dropout
    model.add(Dropout(model_dict['drop_p']))
    # Seventh Layer - First Dense Layer (Fully Connected)
    model.add(Dense(100, activation='tanh'))
    # Eighth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(50, activation='tanh'))
    # Ninth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(10, activation='tanh'))
    # Output
    model.add(Dense(n_classes, activation='softmax'))

    # Optimizer
    adam = optimizers.Adam() # Default Parameters
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

    export_model(model)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.epoch_size,
              verbose=1, validation_data=(X_valid, y_valid))

    score = model.evaluate(X_valid, y_valid, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()