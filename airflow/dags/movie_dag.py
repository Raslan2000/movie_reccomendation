from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np

# Define the DAG
dag = DAG(
    'movie_recommendation_pipeline',
    description='A DAG for processing movie data and training a recommendation model',
    schedule_interval='@daily',
    start_date=datetime(2022, 1, 1),
    catchup=False
)

# Define the tasks
def run_data_ingestion():
    di = DataIngestion()
    train_data_path, test_data_path = di.init_data_ingestion()
    return train_data_path, test_data_path

def run_data_transformation(ti):
    train_data_path, test_data_path = ti.xcom_pull(task_ids='data_ingestion')
    dt = DataTransformation()
    results = dt.initiate_data_transformation(train_data_path, test_data_path)
    train_arr, test_arr, num_users, num_movies, preprocessor_path = results
    return (train_arr.tolist(), test_arr.tolist(), num_users, num_movies, preprocessor_path)

def run_model_training(ti):
    train_arr, test_arr, num_users, num_movies, _ = ti.xcom_pull(task_ids='data_transformation')
    train_arr = np.array(train_arr)  # Convert back to numpy array if necessary
    test_arr = np.array(test_arr)
    mt = ModelTrainer()
    r2_score = mt.initiate_model_trainer(train_arr, test_arr, num_users, num_movies)
    print(f"Model R2 Score: {r2_score}")

# Set up the PythonOperators
data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag,
)

data_transformation = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    provide_context=True,
    dag=dag,
)

model_training = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    provide_context=True,
    dag=dag,
)

# Define dependencies
data_ingestion >> data_transformation >> model_training




