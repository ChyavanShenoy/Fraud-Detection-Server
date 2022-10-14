from gc import callbacks
from matplotlib import test
from matplotlib.dates import MO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Flatten, Dense
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import pickle
from keras.models import Model
import glob

SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
input_ = (SIZE, SIZE, 3)
output_ = 2
MODEL_PATH = './models/model.h5'

test_dir = './data/test_train/test/'
train_dir = './data/test_train/train/'
train_data_names = []
test_data_names = []
train_labels = []
test_labels = []
train_data = []
test_data = []


# Load the model
model = load_model(MODEL_PATH)


def fetch_data():
    # Fetch data from the directory and store in pickle file
    global train_data
    global train_labels
    global test_data
    global test_labels
    for i in os.listdir(train_dir):
        for j in os.listdir(os.path.join(train_dir, i)):
            # img = cv2.imread(os.path.join(train_dir, i, j))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (SIZE, SIZE))
            # train_data.append(img)
            # train_labels.append(i)

            train_data_names.append(j)
            img = cv2.imread(j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (SIZE, SIZE))
            train_data.append(img)

    for i in os.listdir(test_dir):
        for j in os.listdir(os.path.join(test_dir, i)):
            # img = cv2.imread(os.path.join(test_dir, i, j))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (SIZE, SIZE))
            # test_data.append(img)
            # test_labels.append(i)

            test_data_names.append(j)
            img = cv2.imread(j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (SIZE, SIZE))
            test_data.append(img)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # Save the data in pickle file
    with open('./data/train_data.pickle', 'wb') as f:
        pickle.dump(train_data, f)
    with open('./data/train_labels.pickle', 'wb') as f:
        pickle.dump(train_labels, f)
    with open('./data/test_data.pickle', 'wb') as f:
        pickle.dump(test_data, f)
    with open('./data/test_labels.pickle', 'wb') as f:
        pickle.dump(test_labels, f)

    # Resizing the images
    train_data = train_data.reshape(-1, SIZE, SIZE, 3)
    test_data = test_data.reshape(-1, SIZE, SIZE, 3)


def train_model():
    # Load the data from pickle file
    with open('./data/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/train_labels.pickle', 'rb') as f:
        train_labels = pickle.load(f)
    with open('./data/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    with open('./data/test_labels.pickle', 'rb') as f:
        test_labels = pickle.load(f)

    # Train the model
    base_model = applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=input_)
    model = Sequential()
    data_augmentation = keras.Sequential(
        [layers.experimental.preprocessing.RotationRange(0.1)])
    model.add(base_model)
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_, activation='softmax'))

    model = Model(inputs=model.input, outputs=model.output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=1)
    early_stop = [earlyStopping]
    progess = model.fit(train_data, train_labels, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, callbacks=early_stop, validation_split=0.1)
    acc = progess.history['accuracy']
    val_acc = progess.history['val_accuracy']
    loss = progess.history['loss']
    val_loss = progess.history['val_loss']
    epochs = range(len(acc))

    # Save the model
    model.save('./models/model.h5')

    # Plot the accuracy and loss
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    # Evaluate the model
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def train_with_image(path):
    global train_data
    global test_data
    global train_labels
    global test_labels
    # fetch all files from the path
    files = glob.glob(path + '/*')

    # divide the files into train and test
    train_data_names = files[:int(len(files) * 0.8)]
    test_data_names = files[int(len(files) * 0.2):]

    # load the images
    for per in train_data_names:
        img = cv2.imread(per)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        train_data.append(img)
        if per[-1]=='g':
            train_labels.append(np.array(1))
        else:
            train_labels.append(np.array(0))

    for per in test_data_names:
        img = cv2.imread(per)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        test_data.append([img])
        if per[-1]=='g':
            test_labels.append(np.array(1))
        else:
            test_labels.append(np.array(0))

    print("Data reading complete")
    train_data = np.array(train_data)/255.0

    test_data = np.array(test_data)/255.0
    train_labels = np.array(train_labels)

    test_labels = np.array(test_labels)

    print('=========================================================')
    print("Converted to numpy array")
    print('=========================================================')

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Resizing the images
    train_data = train_data.reshape(-1, SIZE, SIZE, 3)
    test_data = test_data.reshape(-1, SIZE, SIZE, 3)

    print("=====================================================================")
    print(F"\nTrain data shape: {train_data.shape}")
    print("=====================================================================")
    print(F"\nTest data shape: {test_data.shape}")
    print("=====================================================================")
    print(F"\nTrain data array: {train_data}")
    print("=====================================================================")
    print(F"\nTest data array: {test_data}")
    print("=====================================================================")

    # convert the data into numpy array
    # train_data = np.array(train_data)
    # train_data = train_data.reshape(-1, SIZE, SIZE, 3)
    # test_data = np.array(test_data)
    # test_data = test_data.reshape(-1, SIZE, SIZE, 3)
    print("Test and train data converted to numpy array")

    # Train the model
    base_model = applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=input_)
    model = Sequential()
    data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
    model.add(base_model)
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_, activation='softmax'))
    model = Model(inputs=model.input, outputs=model.output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=1)
    early_stop = [earlyStopping]
    progess = model.fit(train_data, train_labels, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, callbacks=early_stop, validation_split=0.1)
    acc = progess.history['accuracy']
    val_acc = progess.history['val_accuracy']
    loss = progess.history['loss']
    val_loss = progess.history['val_loss']
    epochs = range(len(acc))

    # Save the model
    model.save('./models/model.h5')

    # Plot the accuracy and loss
