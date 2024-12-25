import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import tensorflow as tf
import tensorflow_datasets as tfds
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def preprocess_function(self, example):
        return {
        'movie_title': tf.strings.unicode_decode(example['movie_title'], 'UTF-8'),
        'user_id': tf.strings.unicode_decode(example['user_id'], 'UTF-8'),
        'rating': example['user_rating']
    }
    
    def dataset_to_dataframe(self, dataset):
        records = [tfds.as_numpy(record) for record in dataset]
        return pd.DataFrame(records)

    def init_data_ingestion(self):
        logging.info('Entered the Data ingestion method')
        try:
            ratings = tfds.load("movielens/100k-ratings", split="train")
            ratings = ratings.map(self.preprocess_function)
            logging.info('Reading the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df = self.dataset_to_dataframe(ratings)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path                
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.init_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    """modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))"""