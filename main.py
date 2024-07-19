import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = './train.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Checking for missing values
print(data.isnull().sum())

# Visualizing missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
plt.show()

# Handling missing values (example: filling Age with median)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

# Converting categorical variables to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Display the first few rows of the modified dataset
print(data.head())

# Splitting the dataset into training and testing sets
X = data.drop(['Survived', 'Name', 'Ticket'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the shapes of the training and testing sets
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')

# Function for simple random sampling
def simple_random_sampling(data, n):
    return data.sample(n)

# Function for stratified sampling
def stratified_sampling(data, stratify_col, n):
    stratified_sample = data.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(int(np.rint(n*len(x)/len(data)))))
    return stratified_sample

# Simple random sampling example
simple_random_sample = simple_random_sampling(data, 100)
print(simple_random_sample.head())

# Stratified sampling example
stratified_sample = stratified_sampling(data, 'Pclass', 100)
print(stratified_sample.head())

# Visualizing the distribution of the sampled data
plt.figure(figsize=(12, 6))
sns.histplot(data['Age'], bins=30, kde=False, label='Original Data')
sns.histplot(simple_random_sample['Age'], bins=30, kde=False, color='red', label='Simple Random Sample')
sns.histplot(stratified_sample['Age'], bins=30, kde=False, color='green', label='Stratified Sample')
plt.legend()
plt.title('Age Distribution')
plt.show()
