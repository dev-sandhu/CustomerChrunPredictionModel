import streamlit as st
import pandas as pd
import pickle
import boto3
from io import BytesIO

# AWS S3 credentials
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'my-airflow-model-bucket'
MODEL_KEY = 'model.pkl'  # The key under which your model is stored in S3

# Load the model from S3
def load_model():
    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    model_file = BytesIO()
    s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
    model_file.seek(0)
    model = pickle.load(model_file)
    return model

# Load model
model = load_model()

# Streamlit UI
st.title('Prediction Application')

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data)
    
    # Make predictions
    predictions = model.predict(input_data)
    st.write("Predictions:")
    st.write(predictions)

    # Option to download predictions
    output = pd.DataFrame(predictions, columns=['Prediction'])
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Predictions",
                       data=csv,
                       file_name='predictions.csv',
                       mime='text/csv')
