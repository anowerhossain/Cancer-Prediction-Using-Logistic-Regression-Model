# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Exercise 1 - Load the cancer dataset
cancer_data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/cancer.csv"
cancer_df = pd.read_csv(cancer_data_url)

# Exercise 2 - Identify the target column and the data columns
target_column = cancer_df["diagnosis"]
feature_columns = cancer_df[['radius_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean']]

# Exercise 3 - Build and Train a new classifier
cancer_classifier = LogisticRegression()
cancer_classifier.fit(feature_columns, target_column)

# Exercise 4 - Evaluate the model
print(cancer_classifier.score(feature_columns, target_column))  # Output: 0.8963093145869947

# Exercise 5 - Find out if a tumor is cancerous
prediction = cancer_classifier.predict([[13.45, 86.6, 555.1, 0.1022, 0.08165, 0.03974, 0.1638]])
print(prediction)  # Output: ['Benign']
