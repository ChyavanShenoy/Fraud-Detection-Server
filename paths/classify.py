import glob
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from shared import delete_files, get_files

SIZE = 224
TESTCLASSIFY = './temp/test'

genuine = []
forged = []

model = load_model('./models/model.h5')


def get_accuracy(genuine, forged):
    genuine_count = len(genuine)
    forged_count = len(forged)
    print('Genuine signatures: ', genuine_count)
    print('Forged signatures: ', forged_count)
    total = genuine_count + forged_count
    accuracy = genuine_count/total
    return accuracy


def classify_signature(path):
    for data in glob.glob(path+'/*.*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = np.array(img)/255.0
        img = img.reshape(-1, SIZE, SIZE, 3)
        pred = model.predict(img)
        pred = np.argmax(pred, axis=1)
        #  classify if signature is genuine or forged
        if pred == 1:
            forged.append(data)
        else:
            genuine.append(data)
    return


def main(path):
    genuine.clear()
    forged.clear()
    get_files(url=path, dest=TESTCLASSIFY)
    classify_signature(TESTCLASSIFY)
    accuracy = get_accuracy(genuine, forged)
    delete_files(folder=TESTCLASSIFY)
    return {
        'test-results':
        {

            'accuracy': accuracy,
            'genuine-signatures': genuine,
            'forged-signatures': forged,
            'total-signatures': len(genuine) + len(forged),
            'genuine-signatures-count': len(genuine),
            'forged-signatures-count': len(forged)
        }

    }
