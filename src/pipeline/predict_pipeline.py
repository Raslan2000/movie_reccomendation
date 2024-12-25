import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import numpy as np


class PredictPipeline:
    def __init__(self):
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocesor.pkl')
    def predict(self,features):
        try:
            print("Before Loading")
            model=load_object(file_path=self.model_path)
            preprocessor=load_object(file_path=self.preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, user_ids, movie_ids):
        self.user_ids = user_ids
        self.movie_ids = movie_ids

    def get_data_as_arrays(self):
        try:
            data_dict = {
                "user_id": self.user_ids,
                "movie_id": self.movie_ids
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)