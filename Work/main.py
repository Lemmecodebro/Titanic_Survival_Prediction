import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

# Descriptive statistics
print(train_df.describe())

# Check for missing values
print(train_df.isnull().sum())

# Visualize the distribution of the target variable
sns.countplot(x='Survived', data=train_df)
plt.title('Distribution of Survived')
plt.show()

# Visualize the distribution of Age
sns.histplot(train_df['Age'].dropna(), bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()

# Visualize the distribution of Fare
sns.histplot(train_df['Fare'].dropna(), bins=30, kde=True)
plt.title('Distribution of Fare')
plt.show()

# Drop non-numeric columns before calculating the correlation matrix
numeric_train_df = train_df.select_dtypes(include=[np.number])

# Check correlation matrix
corr_matrix = numeric_train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

# Visualize survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Pclass')
plt.show()

# Visualize survival rate by Embarked
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Embarked')
plt.show()

# Handle missing values in train data by filling them with median or mode
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())

# Handle missing values in test data similarly
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Store PassengerId for submission
passenger_ids = test_df['PassengerId']

# Drop columns that won't be used for prediction
train_df = train_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Convert categorical variables to numerical using one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target variable for training
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

# Use the test set features for prediction
X_test = test_df.copy()

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Prepare submission DataFrame with PassengerId and predicted survival
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_pred
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

# Split the original train data into a new train and validation set for evaluation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model again on the new training split
model.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model's accuracy on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy}')

# Create a confusion matrix and visualize it
conf_matrix = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print a detailed classification report
report = classification_report(y_val, y_val_pred)
print(report)

# Get the coefficients of the features from the logistic regression model
# Convert X_train_split back to DataFrame if it is a numpy array
if isinstance(X_train_split, np.ndarray):
    feature_names = train_df.drop('Survived', axis=1).columns
    X_train_split = pd.DataFrame(X_train_split, columns=feature_names)

coefficients = pd.DataFrame(model.coef_.flatten(), X_train_split.columns, columns=['Coefficient'])
print(coefficients)

# Analyze feature importance using permutation importance method
perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)

# Create a DataFrame to show feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot the feature importance for visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
