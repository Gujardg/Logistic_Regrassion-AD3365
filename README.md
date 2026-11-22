💼 Employee Salary Classification Using Logistic Regression

An end-to-end Machine Learning project demonstrating Logistic Regression to classify employees into Low Salary and High Salary categories based on their years of work experience using Python and Scikit-Learn.

📋 Overview

This project implements a Logistic Regression model to predict whether an employee belongs to a Low Salary or High Salary category based on their years of professional experience.

The model uses the Salary Dataset and converts the continuous salary values into two categories using the median salary as the classification threshold.

By learning the relationship between years of experience and salary categories, the model predicts the probability of an employee belonging to the High Salary category.

The project covers the complete Machine Learning workflow, including data loading, preprocessing, feature scaling, model training, classification, probability prediction, confusion matrix analysis and model evaluation.

📊 Dataset: Employee Salary Data

The project uses the Salary_Data_Linear_Regression.csv dataset.

The dataset contains 30 records and 2 original columns.

Dataset Features

Feature

Description

YearsExperience

Number of years of professional work experience

Salary

Corresponding employee salary

Dataset Information

Property

Value

Total Records

30

Original Features

2

Input Feature

YearsExperience

Target Variable

SalaryCategory

Missing Values

0

Training Records

24

Testing Records

6

Target Variable: SalaryCategory

The original Salary column is converted into a binary classification target using the median salary.

0 – Low Salary: Salary is below the median salary.

1 – High Salary: Salary is equal to or above the median salary.

The median salary is used as the classification threshold.

🧠 Methodology: Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

Unlike Linear Regression, which predicts continuous numerical values, Logistic Regression predicts the probability of an observation belonging to a particular class.

Mathematical Formula

The Logistic Regression model uses the sigmoid function:

[
P(Y=1|X)=\frac{1}{1+e^{-z}}
]

Where:

[
z=b_0+b_1X
]

Where:

P(Y=1|X) = Probability of High Salary.

X = Years of Experience.

b₀ = Intercept.

b₁ = Model coefficient.

e = Euler's mathematical constant.

Classification Rule

Probability

Predicted Category

Probability below 0.5

Low Salary (0)

Probability equal to or above 0.5

High Salary (1)

Feature Scaling

The project uses StandardScaler() to standardize the YearsExperience feature before training the Logistic Regression model.

Standardization helps the model work with appropriately scaled input values.

📈 Project Pipeline

The complete implementation is available in Logistic_Regrassion.ipynb.

Library Importing: Imported NumPy, Pandas, Matplotlib and Scikit-Learn libraries.

Data Loading: Uploaded and loaded the salary dataset using Pandas.

Data Exploration: Examined dataset dimensions, information, statistical summary and missing values using shape, info(), describe() and isnull().sum().

Target Creation: Created a new SalaryCategory column using the median salary as the classification threshold.

Data Visualization: Created a scatter plot to visualize the relationship between experience and salary categories.

Feature and Target Selection: Selected YearsExperience as the input feature (X) and SalaryCategory as the target variable (y).

Train-Test Split: Divided the dataset into training (80%, 24 records) and testing (20%, 6 records) using train_test_split().

Feature Scaling: Applied StandardScaler() to standardize the training and testing features.

Model Creation: Initialized the Logistic Regression algorithm using LogisticRegression().

Model Training: Trained the model using model.fit(X_train, y_train).

Classification Prediction: Predicted salary categories using model.predict().

Probability Prediction: Calculated class probabilities using model.predict_proba().

Model Evaluation: Evaluated performance using Accuracy, Precision, Recall and F1 Score.

Confusion Matrix: Generated and visualized the confusion matrix to analyze classification performance.

Classification Report: Generated a detailed classification report.

Probability Visualization: Plotted the Logistic Regression probability curve.

New Data Prediction: Predicted the salary category for employees with 5 and 7 years of experience.

Model Saving: Saved and loaded the trained model and scaler using Pickle.

📊 Performance Results

The model was evaluated using the testing dataset containing 6 unseen records.

Evaluation Metrics

Metric

Value

Interpretation

Accuracy

1.00

100% correct predictions

Precision

1.00

100% precision for the positive class

Recall

1.00

100% recall for the positive class

F1 Score

1.00

Perfect F1 score on the test set

Confusion Matrix

The model generated the following confusion matrix:

[
\begin{bmatrix}
3 & 0\
0 & 3
\end{bmatrix}
]

Actual / Predicted

Low Salary (0)

High Salary (1)

Low Salary (0)

3

0

High Salary (1)

0

3

Model Performance Summary

True Positive (TP): 3

True Negative (TN): 3

False Positive (FP): 0

False Negative (FN): 0

Key Result: The Logistic Regression model correctly classified all 6 testing records and achieved 100% accuracy on the test dataset.

Note: The test dataset contains only 6 records. Therefore, these results should not be interpreted as a guarantee of 100% accuracy on new or larger datasets.

🔍 New Data Prediction

The trained model was tested with new experience values.

Prediction for 5 Years of Experience

Output

Result

Years of Experience

5

Predicted Category

Low Salary (0)

Low Salary Probability

64.22%

High Salary Probability

35.78%

Prediction for 7 Years of Experience

Output

Result

Years of Experience

7

Predicted Category

High Salary (1)

The model successfully demonstrates how Logistic Regression can classify new employee records into salary categories.

🛠️ Tech Stack

Technology

Purpose

Python 3.x

Programming language

NumPy

Numerical calculations

Pandas

Data manipulation and analysis

Matplotlib

Data visualization

Scikit-Learn

Machine Learning and model evaluation

StandardScaler

Feature standardization

Pickle

Saving and loading trained models

Google Colab / Jupyter Notebook

Development environment

📦 Getting Started

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git

cd YOUR-REPOSITORY-NAME

2. Install Required Libraries

pip install numpy pandas matplotlib scikit-learn notebook

3. Dataset Setup

Place the Salary_Data_Linear_Regression.csv file in the project directory.

4. Run the Notebook

jupyter notebook Logistic_Regrassion.ipynb

You can also open the notebook directly using Google Colab.

📁 Project Structure

Logistic-Regression-Salary-Classification/
│
├── Logistic_Regrassion.ipynb
│
├── Salary_Data_Linear_Regression.csv
│
├── logistic_model.pkl
│
├── scaler.pkl
│
└── README.md

Note: The model and scaler Pickle files are generated when you execute the model-saving section of the notebook.

🎯 Learning Outcomes

Through this project, I learned:

Fundamentals of Supervised Machine Learning.

Understanding Logistic Regression.

Difference between Regression and Classification.

Converting continuous values into binary classes.

Feature scaling using StandardScaler.

Training and testing classification models.

Understanding the Sigmoid Function.

Predicting class probabilities.

Evaluating classification models.

Understanding Confusion Matrix, Precision, Recall and F1 Score.

Saving and loading Machine Learning models using Pickle.

📝 Conclusion

This project demonstrates the practical implementation of Logistic Regression for employee salary classification using years of experience.

The model successfully classified all 6 test records and achieved 100% accuracy, precision, recall and F1 score on the test dataset.

Through this project, I gained practical knowledge of data preprocessing, feature scaling, classification, probability prediction, model evaluation and model serialization using Python and Scikit-Learn.

👩‍💻 GitHub Submission

Faculty GitHub ID: https://github.com/complexiota

Student GitHub Repository: https://github.com/Gujardg/Logistic_Regrassion-AD3365

Project: Employee Salary Classification Using Logistic Regression

Algorithm: Logistic Regression

Domain: Machine Learning

Programming Language: Python

📄 License

This project is created for educational and academic purposes.
