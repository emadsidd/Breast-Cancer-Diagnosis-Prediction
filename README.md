# Breast Cancer Diagnosis Prediction
Overview
This project focuses on predicting breast cancer diagnoses (malignant or benign) based on features computed from fine needle aspirate (FNA) images of breast masses. The dataset, obtained from the UCI Machine Learning Repository, contains ten real-valued features for each cell nucleus, including characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

Dataset
•	Source: UCI Machine Learning Repository
•	Attributes: ID number, Diagnosis (Malignant or Benign), and 30 computed features.
•	Class Distribution: 357 benign, 212 malignant.

Libraries and Models
The project uses Python with popular machine learning libraries such as pandas, scikit-learn, and seaborn. Three models—Support Vector Machine (SVM), Logistic Regression, and Random Forest—are employed for classification.

Key Steps
1.	Data Analysis and Preprocessing:
•	Exploratory data analysis, visualization, and handling missing values.
•	Mapping diagnosis labels to numerical values and dropping unnecessary columns.
2.	Feature Selection:
•	Calculating and visualizing the correlation matrix.
•	Dropping features with weak correlations based on heatmap analysis.
3.	Model Training and Evaluation:
•	Training SVM, Logistic Regression, and Random Forest classifiers.
•	Hyperparameter tuning using GridSearchCV.
•	Evaluation metrics include accuracy, precision, recall, F1 score, sensitivity, specificity, and AUC-ROC.
4.	Cross-Validation:
•	Utilizing cross-validation for robust model assessment.

Contributions:
Originally a group project, additional work was contributed individually, focusing on enhancing the project with the inclusion of the Random Forest model and additional evaluation metrics.
