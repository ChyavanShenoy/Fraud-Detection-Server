# startup prechecks and loading of models

import os
from paths import classify, train

MODEL_PATH = './models/model.h5'


# Load the model
def check_models():
    if not os.path.exists("./models/model.h5"):
        print("Model does not exist, training model from database")
        train.train_model()
    else:
        print("Model exist")
