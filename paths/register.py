import os
from shared.io import get_files, delete_files
from .train import train_with_image
from tensorflow.keras.models import load_model

TRAIN_PATH = './temp/train/genuine'
MODEL_PATH = './models/'
MODEL_NAME = 'model.h5'


def main(path):
    # get in the files from the path provided
    get_files(url=path, dest=TRAIN_PATH)
    train_with_image(path=TRAIN_PATH)
