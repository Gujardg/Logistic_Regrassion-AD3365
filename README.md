# 💼 Employee Salary Classification Using Logistic Regression

An end-to-end Machine Learning project implementing **Logistic Regression** to classify employees into Low Salary and High Salary categories based on their years of work experience using Python and Scikit-Learn.

---

## 📋 1. Project Overview

This project demonstrates the implementation of a Logistic Regression classification model to predict employee salary categories based on years of professional experience.

The original salary dataset contains continuous salary values. These values are converted into two categories using the median salary as the classification threshold.

The model learns the relationship between years of experience and salary categories and predicts whether an employee belongs to the Low Salary or High Salary category.

This project covers the complete Machine Learning workflow, including data preprocessing, feature scaling, model training, classification, probability prediction, confusion matrix analysis and model evaluation.

---

## 📊 2. Dataset: Employee Salary Data

**Dataset Name:** `Salary_Data_Linear_Regression.csv`

The dataset contains **30 records and 2 original features**.

### Dataset Features

| Feature | Description |
| :--- | :--- |
| `YearsExperience` | Number of years of professional work experience |
| `Salary` | Corresponding employee salary |

### Dataset Summary

| Property | Value |
| :--- | :--- |
| Total Records | 30 |
| Original Features | 2 |
| Input Feature | YearsExperience |
| Target Variable | SalaryCategory |
| Missing Values | 0 |
| Training Samples | 24 |
| Testing Samples | 6 |
| Train-Test Ratio | 80:20 |

### Target Variable: SalaryCategory

The original Salary column is converted into a binary classification variable using the median salary.

The median salary of the dataset is **82,225.50**.

| Category | Label | Condition |
| :--- | :---: | :--- |
| Low Salary | 0 | Salary below median |
| High Salary | 1 | Salary equal to or above median |

This converts the original regression-style salary data into a binary classification problem.

---

## 🧠 3. Methodology: Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

Unlike Linear Regression, which predicts continuous numerical values, Logistic Regression estimates the probability of an observation belonging to a particular class.

### Mathematical Formula

The Logistic Regression algorithm uses the Sigmoid Function:

\[
P(Y=1|X)=\frac{1}{1+e^{-z}}
\]

Where:

\[
z=b_0+b_1X
\]

**Parameters:**

- \(P(Y=1|X)\): Probability of High Salary.
- \(X\): Years of Experience.
- \(b_0\): Model intercept.
- \(b_1\): Model coefficient.
- \(e\): Euler's mathematical constant.

### Classification Rule

The predicted category is determined using a probability threshold of 0.5.

| Probability | Predicted Category |
| :--- | :--- |
| Below 0.5 | Low Salary (0) |
| Equal to or above 0.5 | High Salary (1) |

### Feature Scaling

The project uses `StandardScaler()` from Scikit-Learn to standardize the YearsExperience feature.

Standardization transforms the input feature into a common scale, helping the Logistic Regression model during training.

---

## ⚙️ 4. Project Workflow

The complete implementation is available in [`Logistic_Regrassion.ipynb`](./Logistic_Regrassion.ipynb).

The following steps were performed:

1. **Library Importing:** Imported NumPy, Pandas, Matplotlib and Scikit-Learn libraries.
2. **Data Loading:** Loaded the salary dataset using Pandas.
3. **Data Exploration:** Examined dataset dimensions, information, statistical summary and missing values.
4. **Data Preprocessing:** Created a new `SalaryCategory` column using the median salary.
5. **Data Visualization:** Created a scatter plot to visualize the relationship between years of experience and salary categories.
6. **Feature and Target Selection:** Selected `YearsExperience` as the independent variable (X) and `SalaryCategory` as the dependent variable (y).
7. **Train-Test Split:** Divided the dataset into 80% training data and 20% testing data using `train_test_split()` with stratified sampling.
8. **Feature Scaling:** Applied `StandardScaler()` to standardize the training and testing features.
9. **Model Creation:** Initialized the Logistic Regression model using `LogisticRegression()`.
10. **Model Training:** Trained the model using `model.fit(X_train, y_train)`.
11. **Classification Prediction:** Predicted salary categories using `model.predict()`.
12. **Probability Prediction:** Calculated class probabilities using `model.predict_proba()`.
13. **Model Evaluation:** Evaluated the model using Accuracy, Precision, Recall and F1 Score.
14. **Confusion Matrix:** Generated and visualized the confusion matrix.
15. **Classification Report:** Generated a detailed classification report.
16. **Probability Visualization:** Plotted the Logistic Regression probability curve.
17. **New Data Prediction:** Predicted salary categories for employees with 5 and 7 years of experience.
18. **Model Serialization:** Saved and loaded the trained model and scaler using Python Pickle.

---

## 📈 5. Model Performance Results

The model was evaluated using the testing dataset containing 6 unseen records.

### Evaluation Metrics

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| Accuracy | 1.00 | 100% correct predictions |
| Precision | 1.00 | 100% precision |
| Recall | 1.00 | 100% recall |
| F1 Score | 1.00 | 100% F1 Score |

### Confusion Matrix

The model generated the following confusion matrix:

\[
\begin{bmatrix}
3 & 0\\
0 & 3
\end{bmatrix}
\]

| Actual / Predicted | Low Salary (0) | High Salary (1) |
| :--- | :---: | :---: |
| Low Salary (0) | 3 | 0 |
| High Salary (1) | 0 | 3 |

### Confusion Matrix Interpretation

- **True Positive (TP):** 3
- **True Negative (TN):** 3
- **False Positive (FP):** 0
- **False Negative (FN):** 0

### Key Result

The Logistic Regression model correctly classified all 6 testing records and achieved **100% accuracy** on the test dataset.

**Note:** The testing dataset contains only 6 records. Therefore, the results should not be interpreted as a guarantee of perfect performance on larger or unseen datasets.

---

## 🔍 6. Prediction on New Data

The trained Logistic Regression model was tested with new employee experience values.

### Prediction 1: Employee with 5 Years of Experience

| Parameter | Result |
| :--- | :--- |
| Years of Experience | 5 |
| Predicted Category | Low Salary (0) |
| Low Salary Probability | 64.22% |
| High Salary Probability | 35.78% |

### Prediction 2: Employee with 7 Years of Experience

| Parameter | Result |
| :--- | :--- |
| Years of Experience | 7 |
| Predicted Category | High Salary (1) |

These predictions demonstrate how the trained Logistic Regression model classifies new employee records based on their work experience.

---

## 🛠️ 7. Technologies Used

| Technology | Purpose |
| :--- | :--- |
| Python 3.x | Programming language |
| NumPy | Numerical calculations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Scikit-Learn | Machine Learning model development |
| StandardScaler | Feature standardization |
| Pickle | Model saving and loading |
| Google Colab / Jupyter Notebook | Development environment |

---

## 📦 8. Installation and Execution

Follow these steps to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Gujardg/Logistic_Regrassion-AD3365.git

cd Logistic_Regrassion-AD3365
```

### Step 2: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

### Step 3: Dataset Setup

Make sure the following dataset is available in your project directory:

```text
Salary_Data_Linear_Regression.csv
```

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook Logistic_Regrassion.ipynb
```

Alternatively, open the notebook using Google Colab and upload the required CSV dataset when prompted.

---

## 📁 9. Project Structure

```text
Logistic_Regrassion-AD3365/
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
```

**File Description:**

- `Logistic_Regrassion.ipynb` – Complete implementation of the Logistic Regression model.
- `Salary_Data_Linear_Regression.csv` – Dataset used for training and testing.
- `logistic_model.pkl` – Saved trained Logistic Regression model.
- `scaler.pkl` – Saved StandardScaler object.
- `README.md` – Project documentation.

*Note: The Pickle files are generated by executing the model-saving section of the notebook.*

---

## 🎯 10. Learning Outcomes

Through this project, I gained practical knowledge of:

- Fundamentals of Supervised Machine Learning.
- Understanding Logistic Regression.
- Difference between Linear Regression and Logistic Regression.
- Converting continuous data into binary classes.
- Feature scaling using StandardScaler.
- Training and testing classification models.
- Understanding the Sigmoid Function.
- Predicting class probabilities.
- Evaluating models using Accuracy, Precision, Recall and F1 Score.
- Understanding and interpreting a Confusion Matrix.
- Saving and loading trained models using Pickle.

---

## 📝 11. Conclusion

This project successfully demonstrates the implementation of Logistic Regression for employee salary classification using years of experience.

The trained model achieved 100% accuracy, precision, recall and F1 score on the test dataset containing 6 records.

The project provided practical experience in data preprocessing, feature scaling, classification, probability prediction, model evaluation and model serialization using Python and Scikit-Learn.

---

## 👩‍💻 12. GitHub Submission

**Faculty GitHub ID:** [complexiota](https://github.com/complexiota)

**Student GitHub Repository:** [Logistic Regression – AD3365](https://github.com/Gujardg/Logistic_Regrassion-AD3365)

| Project Details | Information |
| :--- | :--- |
| Project Name | Employee Salary Classification |
| Algorithm | Logistic Regression |
| Domain | Machine Learning |
| Programming Language | Python |
| Repository Visibility | Public |

---

## 📄 13. License

This project is created for educational and academic purposes.
