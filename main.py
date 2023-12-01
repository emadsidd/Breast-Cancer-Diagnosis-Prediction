# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Loading the cancer dataset
df = pd.read_csv('data.csv')

# Analyzing the dataset
print(df)

# Checking the number of rows and columns
num_rows = df.shape[0]
num_cols = df.shape[1]
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# Checking for missing values in the dataset
df.isna().sum()

# Dropping unnecessary columns
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
df.head()

# Analyzing features' data types
df.info()

# Analyzing descriptive statistics of the dataset
df.describe()

# Mapping the 'diagnosis' column to numerical values (0 for benign, 1 for malignant)
df['diagnosis'] = df['diagnosis'].map({'B' : 0, 'M' : 1})
df.head()

# Calculating and visualizing the correlation matrix using a heatmap
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=df.columns, yticklabels=df.columns)
plt.show()

# Dropping columns that have weak correlation based on the heatmap analysis
df.drop(['smoothness_mean', 'symmetry_mean','fractal_dimension_mean','texture_se','smoothness_se','compactness_se','concavity_se','symmetry_se','fractal_dimension_worst','fractal_dimension_se'], axis = 1, inplace = True)
df.info()

# Splitting the dataset into features (X) and target variable (y)
X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']

# Splitting the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the feature values using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Support Vector Machine (SVM) classifier
classifier = svm.SVC(kernel='linear')
# Finding the best parameter for C:
param_grid_svm = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search_svm = GridSearchCV(estimator=svm.SVC(kernel='linear'), param_grid=param_grid_svm, scoring='accuracy', cv=5)
grid_search_svm.fit(X_train, y_train)
print("\nBest Parameter for SVM: ", grid_search_svm.best_params_)
print("Best Accuracy for SVM: ", grid_search_svm.best_score_)
classifier = svm.SVC(kernel='linear', C=0.1)
classifier.fit(X_train, y_train)

# Predicting on the test set and evaluating the SVM model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
y_proba = classifier.decision_function(X_test)
auc_roc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
print("\nSVM F1:",f1)
print("SVM Precision:",precision)
print("SVM Accuracy:", accuracy)
print("SVM Sensitivity:", sensitivity)
print("SVM Specificity:", specificity)
print("SVM AUC-ROC:", auc_roc)
# Cross-validation for SVM
classifier = svm.SVC(kernel='linear', C=0.1)
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
mean_score = np.mean(scores)
std_score = np.std(scores)
print("Cross-validation accuracy for SVM: {:.2f} ± {:.2f}".format(mean_score, std_score))


# Training a Logistic Regression model
log_reg = LogisticRegression(max_iter=5000)
# Finding the best parameter for C:
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, scoring='accuracy', cv=5)
grid_search_lr.fit(X_train, y_train)
print("\nBest Parameter for LR: ", grid_search_lr.best_params_)
print("Best Accuracy for LR: ", grid_search_lr.best_score_)
log_reg = LogisticRegression(max_iter=5000, C=1)
log_reg.fit(X_train, y_train)

# Predicting on the test set and evaluating the logistic regression model
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
sensitivity_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
specificity_lr = accuracy_score(y_test, y_pred_lr, normalize=True, sample_weight=None)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
auc_roc_lr = roc_auc_score(y_test, y_proba_lr)
print("\nLR F1:", f1_lr)
print("LR Precision:", precision_lr)
print("LR Accuracy:", accuracy_lr)
print("LR Sensitivity:", sensitivity_lr)
print("LR Specificity:", specificity_lr)
print("LR AUC-ROC:", auc_roc_lr)

# Cross-validation for logistic regression
log_reg = LogisticRegression(max_iter=5000, C=1)
scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
mean_score = np.mean(scores)
std_score = np.std(scores)
print("Cross-validation accuracy for LR: {:.2f} ± {:.2f}".format(mean_score, std_score))

# Training Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=0)
# Finding the best parameters for Random Forest:
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, scoring='accuracy', cv=5)
grid_search_rf.fit(X_train, y_train)
print("\nBest Parameters for RF: ", grid_search_rf.best_params_)
print("Best Accuracy for RF: ", grid_search_rf.best_score_)
rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
rf_classifier.fit(X_train, y_train)

# Predicting on the test set and evaluating the Random Forest model
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
sensitivity_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
specificity_rf = accuracy_score(y_test, y_pred_rf, normalize=True, sample_weight=None)
y_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]
auc_roc_rf = roc_auc_score(y_test, y_proba_rf)

print("\nRandom Forest F1:", f1_rf)
print("Random Forest Precision:", precision_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Sensitivity:", sensitivity_rf)
print("Random Forest Specificity:", specificity_rf)
print("Random Forest AUC-ROC:", auc_roc_rf)

# Cross-validation for Random Forest
rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
scores_rf = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
mean_score_rf = np.mean(scores_rf)
std_score_rf = np.std(scores_rf)
print("Cross-validation accuracy for Random Forest: {:.2f} ± {:.2f}".format(mean_score_rf, std_score_rf))
