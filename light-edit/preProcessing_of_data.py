import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder,StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
data = pd.read_csv("agriculture_dataset.csv")
data.head()
print (data.shape)
print (data.isnull().sum())
print (data.info())
print (data.describe())
p=data['Crop_Type'].value_counts()
q=data['Irrigation_Type'].value_counts()
r=data['Soil_Type'].value_counts()
s= data['Season'].value_counts()
data =data.drop(columns ='Farm_ID')
print(p,q,r,s,data)
num_cols = data.select_dtypes(include =['int64', 'float64']).columns.tolist()
cat_cols = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
print(num_cols)
print(cat_cols)
for column in num_cols:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print('Outlier in: ', column)
    print(outliers[[column]])
    print('Number of outliers: ', len(outliers), '\n')
crop_encoder = LabelEncoder()
irrigation_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
season_encoder = LabelEncoder()
data['Crop_Type'] = crop_encoder.fit_transform(data['Crop_Type'])
data['Irrigation_Type'] = irrigation_encoder.fit_transform(data['Irrigation_Type'])
data['Soil_Type'] = soil_encoder.fit_transform(data['Soil_Type'])
data['Season'] = season_encoder.fit_transform(data['Season'])
print(crop_encoder.classes_)
print(irrigation_encoder.classes_)
len(data.columns)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[num_cols])

# Convert back to DataFrame
data_scaled_standard = pd.DataFrame(data_scaled, columns=num_cols)

# Save preprocessed data
data_scaled_standard.to_csv("preprocessed.csv", index=False)
