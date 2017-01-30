import numpy as np
import sys, os
import pandas as pd
import argparse
import cv2
import pickle
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize, imshow
from sklearn import preprocessing




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

    X_train = np.append(df_input_data.center.values, (df_input_data.right.values, df_input_data.left.values))
    y_train = np.append(df_input_data.steering.values.astype(np.float32),
                        (df_input_data.steering.values.astype(np.float32),
                         df_input_data.steering.values.astype(np.float32)))

    # histogram plottin

    df_input_data.steering.hist()

    return shuffle(X_train, y_train)

def nvidia_model(model):

    """

    https://arxiv.org/pdf/1604.07316v1.pdf

    :param model: keras object
    :return: keras object modeled to nvidia architecture
    """

    model_dict = {'in_shape': (66, 200, 3),  # For tensorflow backend use shape as (row, cols, chan)
                  'nb_filter_1': 24, 'nb_row_1': 5, 'nb_col_1': 5, 'stride_1': (2, 2), 'z_1': 'relu', 'pad_1': 'valid',
                  'nb_filter_2': 36, 'nb_row_2': 5, 'nb_col_2': 5, 'stride_2': (2, 2), 'z_2': 'relu', 'pad_2': 'valid',
                  'nb_filter_3': 48, 'nb_row_3': 5, 'nb_col_3': 5, 'stride_3': (2, 2), 'z_3': 'relu', 'pad_3': 'valid',
                  'nb_filter_4': 64, 'nb_row_4': 2, 'nb_col_4': 2, 'stride_4': (1, 1), 'z_4': 'relu', 'pad_4': 'valid',
                  'nb_filter_5': 64, 'nb_row_5': 2, 'nb_col_5': 2, 'stride_5': (1, 1), 'z_5': 'relu', 'pad_5': 'valid',
                  'drop_p': 0.4}

    # First Layer - 3@66x200 -> 24@31x98
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_1'], nb_row=model_dict['nb_row_1'],
                            nb_col=model_dict['nb_col_1'], border_mode=model_dict['pad_1'],
                            init='normal',
                            input_shape=model_dict['in_shape']))

    model.add(MaxPooling2D(pool_size=(2,2),strides=model_dict['stride_1']))

    model.add(Activation('relu'))
    # Second Layer - 24@31x98 -> 36@14x47
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_2'], nb_row=model_dict['nb_row_2'],
                            nb_col=model_dict['nb_col_2'], border_mode=model_dict['pad_2'],
                            init='normal'
                            ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=model_dict['stride_1']))
    model.add(Activation('relu'))

    # Third Layer - 36@14x47 -> 48@5x22
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_3'], nb_row=model_dict['nb_row_3'],
                            nb_col=model_dict['nb_col_3'], border_mode=model_dict['pad_3'],
                            init='normal'  ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=model_dict['stride_1']))
    model.add(Activation('relu'))

    # Fourth Layer  - 48@5x22 -> 64@3x20
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_4'], nb_row=model_dict['nb_row_4'],
                            nb_col=model_dict['nb_col_4'], border_mode=model_dict['pad_4'],
                            init='normal', subsample=model_dict['stride_4']))

    model.add(Activation('relu'))

    # Fifth Layer
    model.add(Convolution2D(nb_filter=model_dict['nb_filter_5'], nb_row=model_dict['nb_row_5'],
                            nb_col=model_dict['nb_col_5'], border_mode=model_dict['pad_5'],
                            init='normal', subsample=model_dict['stride_5']))

    model.add(Activation('relu'))

    # Flat model - prepare for Dense Layers
    model.add(Flatten())
    model.add(Activation('relu'))

    # Sixth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(model_dict['drop_p']))

    # Seventh Layer - First Dense Layer (Fully Connected)
    model.add(Dense(100, activation='relu'))

    # Eighth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(50, activation='relu'))

    # Ninth Layer - First Dense Layer (Fully Connected)
    model.add(Dense(10, activation='relu'))

    # Output
    model.add(Dense(1))

    # Optimizer

    # adag = optimizers.adagrad(lr=0.001)
    adam = optimizers.adam(lr=0.0001)

    model.compile(loss='mse',
                  optimizer=adam)

    return model

def export_model(model):

    # Model exportation
    # It saves the neural net architecture into
        # Image file model.png to visualization
        # JSON file to model loading
        # H5 file for weights and bias saving

    from keras.utils.visualize_util import plot

    # Plotting model architecture to png file
    plot(model, to_file='model.png')

    # Model architecture to json
    json_string = model.to_json()

    with open('model.json', mode='w') as file:
        file.write(json_string)
        file.close()

    # Model weights to file
    model.save_weights('model.h5')

    return 0


def generator(X_data, y_data, batch_size = 128, scaling_factor = 5):
    # In order to handle large datasets which may not fit in memory, this example loads only a set of images
    # according to user requirements

    X_data, y_data = shuffle(X_data, y_data)

    image_h_target = 66
    image_w_target = 200

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode='nearest')

    while 1:
        for i in range(0, int(len(X_data) / batch_size)):

            offset = i * batch_size

            # create output array with size equal to scaling
            X_out = np.ones(shape=(batch_size * scaling_factor, image_h_target, image_w_target, 3), dtype=np.float32)
            y_out = np.zeros(shape=(batch_size * scaling_factor), dtype=float)

            for j in range(0, batch_size):
                file_path = X_data[offset + j].split('/')[-1]

                im_data = imresize(imread(os.path.join('data/IMG', file_path)), size=(image_h_target, image_w_target, 3))
                #im_data = cv2.cvtColor(im_data, cv2.cv2.COLOR_RGB2HSL)
                # resize images

                image = im_data.astype(dtype='float32')
                image -= np.mean(image, axis=0)
                image /= np.std(image, axis=0)

                if scaling_factor > 1:
                    k = 0
                    #for b in datagen.flow(np.array([image]), np.array([y_data[offset + j]]), batch_size=scaling_factor):
                    X_out[(j * scaling_factor + 0)] = cv2.flip(image, 1)
                    y_out[(j * scaling_factor + 0)] = -np.array([y_data[offset + j]])
                    k += 1
                    # if k == (scaling_factor - 1):
                    X_out[(j * scaling_factor + 1)] = image
                    y_out[(j * scaling_factor + 1)] = np.array([y_data[offset + j]])
                            #break

                else:
                    X_out[j][:] = image

            if scaling_factor > 1:
                yield X_out, y_out
            else:
                yield X_out, y_data[offset: offset + batch_size]

def load_images(X_data):
    image_h_target = 66
    image_w_target = 200

    X_out = np.ones(shape=(len(X_data), image_h_target, image_w_target, 3), dtype=np.float32)

    for i, v in enumerate(X_data):
        file_path = v.split('/')[-1]
        im_data = imresize(imread(os.path.join('data/IMG', file_path)), size=(image_h_target, image_w_target, 3))
        #im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2HSL)
        # resize images

        image = im_data.astype(dtype='float32')
        image -= np.mean(image, axis=0)
        image /= np.std(image, axis=0)

        X_out[i][:] = image


    return X_out


def parse():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument("-b", "--batch_size", default=128, type=int, help='Batch Size')
    parser.add_argument("-e", "--epoch_size", type=int, default=5, help="Number of epochs")
    parser.add_argument("-s", "--batch_scaling", type=int, default=3, help="Scaling factor for batch")
    args = parser.parse_args()
    return args


def main():
    args = parse()

    model = Sequential()

    # ----- Model definition -----
    model = nvidia_model(model)

    if not os.path.exists('model.h5') and not os.path.exists('model.json'):

        # spliting train, validation and test set
        X_train, y_train = read_csv()
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid)

        print("Train/Validation/Test Size: %d/%d/%d" % (len(X_train), len(X_valid), len(X_test)))

        # --- Callbacks ---
        # checkpoint
        checkpoint = ModelCheckpoint('check', monitor='loss', verbose=1, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        # early termination
        early_termination = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto')

        # training data
        model.fit_generator(generator(X_train, y_train, scaling_factor=args.batch_scaling),
                    nb_epoch=args.epoch_size,
                    validation_data=generator(X_valid, y_valid, batch_size=args.batch_size),
                    nb_val_samples=X_valid.size * args.batch_scaling,
                    samples_per_epoch=int((X_train.size / args.batch_size)) * args.batch_size * args.batch_scaling,
                    verbose=1,
                    callbacks=[checkpoint, early_termination])

        export_model(model)

        score = model.evaluate(load_images(X_test), y_test, verbose=1)

        print('Test score:', score)

    else:
        from keras.models import model_from_json
        # Load JSON model
        with open('model.json','r') as f:
            loaded_model = f.read()
            f.close()

        loaded_model = model_from_json(loaded_model)
        print("Model Loaded from File")

        # Load Weights
        print("Wheights loaded from file")
        loaded_model.load_weights('model.h5')

        loaded_model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        model = loaded_model


    # Predict from samples
    # Hand picked images from different classes
    # center_2016_12_01_13_31_13_381.jpg    0
    # center_2016_12_01_13_32_47_293.jpg    0.38709
    # center_2016_12_01_13_33_54_272.jpg    -0.2306556

    images_predict = load_images(np.array(['center_2016_12_01_13_32_47_293.jpg',
                                           'center_2016_12_01_13_31_13_381.jpg',
                                           'center_2016_12_01_13_33_54_272.jpg']))

    predictions = model.predict(images_predict)


    print('predicted/correct: %s/%s' % (predictions[0][0], 0))
    print('predicted/correct: %s/%s' % (predictions[1][0], 0.38709))
    print('predicted/correct: %s/%s' % (predictions[2][0], -0.2306556))



if __name__ == '__main__':
    main()