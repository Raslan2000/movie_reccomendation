import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    


def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)
        
def load_NCFmodel(file_path):
    try:
          return load_model(file_path)
    except Exception as e:
            raise CustomException(e, sys)