import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

x_train = pd.read_csv('https://raw.githubusercontent.com/ssandhu1718/Churn-Data-Hosting/main/X_train.csv')
y_train = pd.read_csv('https://raw.githubusercontent.com/ssandhu1718/Churn-Data-Hosting/main/y_train.csv')

print("Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Check the number of rows in x_train
num_rows_x_train = x_train.shape[0]  # This returns the number of rows

print(f"The number of rows in x_train is: {num_rows_x_train}")

# Remove irrelevant columns
x_train.drop(['Surname', 'CustomerId'], axis=1, inplace=True)

print("Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Check the number of features
num_features = x_train.shape[1]
print(f'The dataset currently has {num_features} features.')

# Replace '?' with NaN to handle missing values
x_train.replace('?', np.nan, inplace=True)

# Check the number of rows in x_train
num_rows_x_train = x_train.shape[0]  # This returns the number of rows

print(f"The number of rows in x_train is: {num_rows_x_train}")

# Display missing values in each column
missing_values = x_train.isnull().sum()
print(f'Missing values in each column:\n{missing_values}')

print(" Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Remove extra spaces and convert categorical values to lowercase
if 'Gender' in x_train.columns:
    x_train['Gender'] = x_train['Gender'].str.strip().str.lower()

print(" Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Check the number of rows in x_train
num_rows_x_train = x_train.shape[0]  # This returns the number of rows

print(f"The number of rows in x_train is: {num_rows_x_train}")

# Check for any non-numeric columns
non_numeric_columns = x_train.select_dtypes(include=['object']).columns
print(f'Non-numeric columns:\n{non_numeric_columns}')

# One-Hot Encode categorical variables
categorical_cols = ['Gender', 'Geography']
x_train = pd.get_dummies(x_train, columns=categorical_cols, drop_first=False)

print(" Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Check if the data is now fully numeric
print("Data types after processing:\n", x_train.dtypes)

# Print the first few rows of the cleansed dataset
print("Cleansed Dataset:")
print(x_train.head())  # This will print the first 5 rows of the dataset

# Check for missing values in y_train
missing_values_y_train = y_train.isnull().sum()

print("Missing values in y_train:")
print(missing_values_y_train)

# Check the number of rows in y_train
num_rows_y_train = y_train.shape[0]  # This returns the number of rows

print(f"The number of rows in y_train is: {num_rows_y_train}")

# Cleansed X_train dataset
print("Cleansed Dataset:")
print(x_train.head())  # This is your cleansed X_train

# Drop the 'CustomerId' column from y_train
y_train_cleaned = y_train.drop(columns=['CustomerId'])

# Print y_train_cleaned (first 5 samples)
print("y_train after removing CustomerId (first 5 samples):")
print(y_train_cleaned.head())

# Combine cleansed X_train with y_train_cleaned along the columns
combined_data = pd.concat([x_train, y_train_cleaned], axis=1)

# Print the combined dataset
print("Combined Dataset:")
print(combined_data.head())

# Export the combined DataFrame to a CSV file
combined_data.to_csv('combined_dataset.csv', index=False)

print("The combined dataset has been exported to 'combined_dataset.csv'.")

# Convert the DataFrame to a JSON string
json_data = combined_data.to_json(orient='records')

# Write the JSON string to a file
with open('output_file.json', 'w') as json_file:
    json_file.write(json_data)

print("CSV has been converted to JSON and saved as 'combined_data.json'")


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train['Exited'], test_size=0.2, random_state=42) 

# Model Selection: Random Forest
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Model Training
model.fit(X_train, y_train)

# Predict on validation data
y_val_pred = model.predict(X_val)


# Print to confirm
print("y_train (first 5 samples after removing CustomerId):")
print(y_train[:5])

# Correct prediction method
y_val_pred = model.predict(X_val)

# Now check the shape again to confirm it's 1D
print(f"Shape of y_val_pred: {y_val_pred.shape}")


# Evaluate the model's performance on validation data
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))