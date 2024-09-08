import streamlit as st
import pandas as pd
import pickle
import boto3
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS to change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f2ecfa; /* Light grey background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
st.title('PredictPro: Customer Churn Prediction Dashboard')

# Sidebar for user inputs and options
st.sidebar.header('Upload and Configure')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.sidebar.write("Uploaded Data Overview:")
    st.sidebar.write(input_data.head())

    # Display basic statistics
    st.header('Data Summary')
    st.write("Below is a summary of the uploaded data:")
    st.write(input_data.describe())

    # Show correlation matrix
    st.subheader('Correlation Matrix')
    correlation = input_data.corr()
    st.write(px.imshow(correlation, text_auto=True))

    # Make predictions
    predictions = model.predict(input_data)
    input_data['Prediction'] = predictions

    # Display predictions
    st.header('Predictions')
    st.write(input_data)

    # Visualization: Donut Chart of Predictions
    st.subheader('Prediction Distribution')
    prediction_counts = input_data['Prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']
    
    fig = go.Figure(data=[go.Pie(labels=prediction_counts['Prediction'], values=prediction_counts['Count'], hole=.3)])
    st.plotly_chart(fig)

    # Visualization: Bar Chart
    st.subheader('Feature Importance Visualization')
    feature_importance = model.feature_importances_
    features = pd.DataFrame({'Feature': input_data.columns[:-1], 'Importance': feature_importance})
    features = features.sort_values(by='Importance', ascending=False)

    fig_bar = px.bar(features, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_bar)

    # Visualization: Trend Over Time
    if 'Date' in input_data.columns:
        st.subheader('Prediction Trend Over Time')
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        trend = input_data.groupby('Date')['Prediction'].mean().reset_index()

        fig_trend = px.line(trend, x='Date', y='Prediction', title='Prediction Trend Over Time')
        st.plotly_chart(fig_trend)
    else:
        print("")

    # Option to download predictions
    output = input_data[['Prediction']].copy()
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Predictions",
                       data=csv,
                       file_name='predictions.csv',
                       mime='text/csv')

    # Allow users to select features to visualize
    st.sidebar.header('Visualization Options')
    selected_features = st.sidebar.multiselect('Select features to visualize', options=input_data.columns[:-1])
    
    if selected_features:
        for feature in selected_features:
            st.subheader(f'Distribution of {feature}')
            fig_feature = px.histogram(input_data, x=feature, title=f'Distribution of {feature}')
            st.plotly_chart(fig_feature)

else:
    st.warning("Please upload a CSV file to proceed.")
