from gc import callbacks
from matplotlib import test
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

test_dir = './data/test_train/test/'
train_dir = './data/test_train/train/'
train_data_names = []
train_labels = []
train_data = []
test_data = []


# Load the model
model = load_model('./models/model.h5')


def fetch_data():
    # Fetch data from the directory and store in pickle file
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in os.listdir(train_dir):
        for j in os.listdir(os.path.join(train_dir, i)):
            img = cv2.imread(os.path.join(train_dir, i, j))
            img = cv2.resize(img, (SIZE, SIZE))
            train_data.append(img)
            train_labels.append(i)
    for i in os.listdir(test_dir):
        for j in os.listdir(os.path.join(test_dir, i)):
            img = cv2.imread(os.path.join(test_dir, i, j))
            img = cv2.resize(img, (SIZE, SIZE))
            test_data.append(img)
            test_labels.append(i)
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
    # fetch all files from the path
    files = glob.glob(path + '/*')

    # divide the files into train and test
    train_data_names = files[:int(len(files) * 0.5)]
    test_data_names = files[int(len(files) * 0.5):]

    # load the images
    for i in train_data_names:
        img = cv2.imread(i)
        img = cv2.resize(img, (SIZE, SIZE))
        train_data.append(img)
    for i in test_data_names:
        img = cv2.imread(i)
        img = cv2.resize(img, (SIZE, SIZE))
        test_data.append(img)

    # convert the data into numpy array
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # Resizing the images
    train_data = train_data.reshape(-1, SIZE, SIZE, 3)
    test_data = test_data.reshape(-1, SIZE, SIZE, 3)

    # Load the model
    model = load_model('./models/model.h5')

    # train the model with the new data
    model.fit(train_data, test_data, batch_size=BATCH_SIZE,
              epochs=EPOCHS, validation_split=0.1)

    # Save the model
    model.save('./models/model.h5')

    # for file in files:
    #     train_data_names.append(file)
    # # load the images
    # for i in train_data_names:
    #     img = cv2.imread(i)
    #     img = cv2.resize(img, (SIZE, SIZE))
    #     train_data.append(img)
    # train_data = np.array(train_data)
    # train_data = train_data.reshape(-1, SIZE, SIZE, 3)

    # partially train model with the new data
    # model.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
    # model.partial_fit(train_data).fileName.text()
    # save the model
    # model.save('./models/model.h5')
