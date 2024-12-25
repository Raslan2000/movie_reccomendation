import os
import sys
from dataclasses import dataclass
import tensorflow as rf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply, Dropout
from tensorflow.keras.optimizers import Adam
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, num_users, num_movies):
        user_input = Input(shape=(1,), name='user_input', dtype='int32')
        movie_input = Input(shape=(1,), name='movie_input', dtype='int32')
        # GMF part
        gmf_user_embedding = Embedding(num_users, 10, input_length=1, name='gmf_user_embedding')(user_input)
        gmf_user_embedding = Flatten()(gmf_user_embedding)

        gmf_movie_embedding = Embedding(num_movies, 10, input_length=1, name='gmf_movie_embedding')(movie_input)
        gmf_movie_embedding = Flatten()(gmf_movie_embedding)

        gmf_vector = Multiply()([gmf_user_embedding, gmf_movie_embedding])

        # MLP part
        mlp_user_embedding = Embedding(num_users, 32, input_length=1, name='mlp_user_embedding')(user_input)
        mlp_user_embedding = Flatten()(mlp_user_embedding)

        mlp_movie_embedding = Embedding(num_movies, 32, input_length=1, name='mlp_movie_embedding')(movie_input)
        mlp_movie_embedding = Flatten()(mlp_movie_embedding)

        mlp_vector = Concatenate()([mlp_user_embedding, mlp_movie_embedding])
        mlp_vector = Dense(64, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)
        mlp_vector = Dense(32, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)
        mlp_vector = Dense(16, activation='relu')(mlp_vector)

        # Combine GMF and MLP parts
        combined_vector = Concatenate()([gmf_vector, mlp_vector])
        outputs = Dense(1, activation='linear')(combined_vector)
        model = Model(inputs=[user_input, movie_input], outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error') 
        return model
    
    def initiate_model_trainer(self,train_array,test_array, num_users, num_movies):
        try:
            model = self.build_model(num_users, num_movies )
            X_train, y_train = [train_array[:, 0], train_array[:, 1]], train_array[:, 2]
            X_test, y_test = [test_array[:, 0], test_array[:, 1]], test_array[:, 2]
            logging.info("Split training and test input data")
            model.fit([X_train[0], X_train[1]], y_train, epochs=10, batch_size=64, validation_split=0.1)
            logging.info("Trained the model")
            logging.info("Evaluating NCF model")
            predictions = model.predict([X_test[0], X_test[1]])
            r2 = r2_score(y_test, predictions.flatten())
            logging.info("Saving NCF model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            return r2
        except Exception as e:
            raise CustomException(e,sys)