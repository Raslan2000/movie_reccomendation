import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocesor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def encode_ids(self, data_frame, feature_columns):
        '''
        Adjust this function to properly handle pandas DataFrame
        '''
        encoding_info = {}
        for feature in feature_columns:
            unique = data_frame[feature].unique()
            mapping = {key: idx for idx, key in enumerate(unique)}
            data_frame[feature] = data_frame[feature].map(mapping)
            encoding_info[feature] = (mapping, len(unique))
        return encoding_info

    
    def get_data_transformer_object(self):
        """
        This function creates and returns a preprocessor object that handles transformations for both user_id and movie_title.
        """
        try:
            # Define the features to be preprocessed
            user_id_feature = ['user_id']
            movie_title_feature = ['movie_title']

            # Define pipelines for individual features
            user_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),  # Encode user IDs as ordinal values
                    ('scaler', StandardScaler(with_mean=False))  # Scale values
                ]
            )

            movie_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),  # Encode movie titles as ordinal values
                    ('scaler', StandardScaler(with_mean=False))  # Scale values
                ]
            )

            logging.info("User ID and Movie Title columns processed separately.")

            # Combine pipelines for both features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('user_pipeline', user_pipeline, user_id_feature),
                    ('movie_pipeline', movie_pipeline, movie_title_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            encoding_info_train = self.encode_ids(train_df, ['user_id', 'movie_title'])
            encoding_info_test = self.encode_ids(test_df, ['user_id', 'movie_title'])
            num_users = encoding_info_train['user_id'][1]  # the number of unique users
            num_movies = encoding_info_train['movie_title'][1]  # the number of unique movies
            logging.info("Reading train and test set")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_df = train_df.drop(columns=['rating'], axis = 1)
            target_feature_train_df = train_df["rating"]
            input_feature_test_df = test_df.drop(columns=['rating'], axis = 1)
            target_feature_test_df = test_df["rating"]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                num_users,
                num_movies,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        

