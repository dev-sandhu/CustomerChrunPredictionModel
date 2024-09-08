from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import pickle
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
}

# Define the DAG
dag = DAG(
    'ai_model_pipeline',
    default_args=default_args,
    description='A simple AI pipeline to train and store model to S3',
    schedule_interval=None,
)

# Task 1: Read the CSV file
def read_csv(**kwargs):
    df = pd.read_csv('/opt/airflow/materials/combined_dataset.csv')
    # Convert DataFrame to a list of dictionaries
    data = df.to_dict(orient='records')
    # Push the serialized data to XCom
    kwargs['ti'].xcom_push(key='data', value=data)

read_csv_task = PythonOperator(
    task_id='read_csv',
    python_callable=read_csv,
    provide_context=True,
    dag=dag,
)

# Task 2: Build the AI Model
def build_model(**kwargs):
    df = kwargs['ti'].xcom_pull(key='data')
    df = pd.DataFrame(df)
    X = df.drop(columns=['Exited'])  # Replace 'target_column' with the actual target column name
    y = df['Exited']  # Replace 'target_column' with the actual target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open('/opt/airflow/dags/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    kwargs['ti'].xcom_push(key='model_path', value='/opt/airflow/dags/model.pkl')

build_model_task = PythonOperator(
    task_id='build_model',
    python_callable=build_model,
    provide_context=True,
    dag=dag,
)

# Task 3: Export the model to S3
def export_to_s3(**kwargs):
    s3 = boto3.client('s3')
    model_path = '/opt/airflow/dags/model.pkl'
    s3.upload_file(model_path, 'my-airflow-model-bucket', 'model.pkl')  # Replace 'your-s3-bucket-name' with your actual S3 bucket name

export_to_s3_task = PythonOperator(
    task_id='export_to_s3',
    python_callable=export_to_s3,
    provide_context=True,
    dag=dag,
)

# Setting task dependencies
read_csv_task >> build_model_task >> export_to_s3_task
