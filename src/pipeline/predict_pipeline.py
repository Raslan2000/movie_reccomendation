import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, load_NCFmodel
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,user_ids, movie_titles):
        try:
            model_path=os.path.join("artifacts","model.keras")
            preprocessor_path=os.path.join('artifacts','preprocesor.pkl')
            print("Before Loading")
            model=load_NCFmodel(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            features = pd.DataFrame({
                "user_id": user_ids,
                "movie_title": movie_titles
            })

            print("Input Features DataFrame:\n", features)
            if features.empty:
                raise ValueError("Input features DataFrame is empty.")
            data_scaled = preprocessor.transform(features)
            user_input = data_scaled[:, 0].toarray()
            movie_input = data_scaled[:, 1].toarray()
            print([user_input, movie_input])
            predictions = model.predict([user_input, movie_input])
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
