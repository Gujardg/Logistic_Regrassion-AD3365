# 💼 Employee Salary Classification Using Logistic Regression

An end-to-end Machine Learning project implementing **Logistic Regression** to classify employees into Low Salary and High Salary categories based on their total working years using Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-Learn.

---

## 📋 1. Project Overview

This project demonstrates the implementation of a Logistic Regression classification model to classify employees into salary categories based on their work experience.

Using the IBM HR Analytics Employee Attrition dataset, the project analyzes employee information and predicts whether an employee belongs to the Low Salary or High Salary category.

The original monthly income values are converted into two categories using the median monthly income as the classification threshold.

The project covers the complete Machine Learning workflow, including data loading, data exploration, preprocessing, data visualization, feature scaling, model training, classification, probability prediction, confusion matrix analysis and model evaluation.

---

## 📊 2. Dataset: IBM HR Analytics Employee Attrition Dataset

**Dataset Name:** `HR-Employee-Attrition.csv`

**Dataset Download Link:**

[Download IBM HR Employee Attrition Dataset](https://raw.githubusercontent.com/OmerFarukMerey/IBM-HR-Analytics-Employee-Attrition-Performance/main/HR-Employee-Attrition.csv)

The original dataset contains **1,470 employee records and 35 columns**.

### Dataset Features Used

| Feature | Description |
| :--- | :--- |
| `TotalWorkingYears` | Total number of years of employee work experience |
| `MonthlyIncome` | Monthly salary of the employee |
| `SalaryCategory` | Newly created binary salary classification |

### Dataset Summary

| Property | Value |
| :--- | :--- |
| Dataset Name | IBM HR Analytics Employee Attrition |
| Total Records | 1,470 |
| Original Columns | 35 |
| Input Feature | TotalWorkingYears |
| Original Salary Column | MonthlyIncome |
| Target Variable | SalaryCategory |
| File Format | CSV |
| Missing Values | Checked during preprocessing |
| Train-Test Ratio | 80:20 |

### Target Variable: SalaryCategory

The original `MonthlyIncome` column is converted into a binary classification variable using the median monthly income.

The median is calculated from the loaded dataset using:

```python
median_salary = df['MonthlyIncome'].median()
```

The salary categories are created as follows:

| Category | Label | Condition |
| :--- | :---: | :--- |
| Low Salary | 0 | MonthlyIncome below median |
| High Salary | 1 | MonthlyIncome equal to or above median |

This converts the original numerical salary values into a binary classification problem.

**Note:** The original dataset also contains an `Attrition` column, but this project does not use it as the prediction target.

---

## 🧠 3. Methodology: Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

It estimates the probability of an observation belonging to a particular class.

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

- **P(Y=1|X):** Probability of High Salary.
- **X:** Total Working Years.
- **b₀:** Model intercept.
- **b₁:** Model coefficient.
- **e:** Euler's mathematical constant.

### Classification Rule

The predicted category is determined using a probability threshold of 0.5.

| Probability | Predicted Category |
| :--- | :--- |
| Below 0.5 | Low Salary (0) |
| Equal to or above 0.5 | High Salary (1) |

### Feature Scaling

The project uses `StandardScaler()` from Scikit-Learn to standardize the `TotalWorkingYears` feature.

Standardization transforms the input feature into a common scale, helping the Logistic Regression model during training.

---

## ⚙️ 4. Project Workflow

The complete implementation is available in [`Logistic_Regrassion.ipynb`](./Logistic_Regrassion.ipynb).

The following steps are performed:

1. **Library Importing:** Imported NumPy, Pandas, Matplotlib, Seaborn and Scikit-Learn libraries.
2. **Data Loading:** Loaded the IBM HR employee dataset using Pandas.
3. **Data Exploration:** Examined dataset dimensions, column names, information, statistical summary and missing values.
4. **Data Preprocessing:** Created a new `SalaryCategory` column using the median of `MonthlyIncome`.
5. **Data Visualization:** Created scatter plots to visualize the relationship between total working years and salary categories.
6. **Feature and Target Selection:** Selected `TotalWorkingYears` as the independent variable (X) and `SalaryCategory` as the dependent variable (Y).
7. **Train-Test Split:** Divided the dataset into 80% training data and 20% testing data using `train_test_split()`.
8. **Feature Scaling:** Applied `StandardScaler()` to standardize the training and testing features.
9. **Model Creation:** Initialized the Logistic Regression model using `LogisticRegression()`.
10. **Model Training:** Trained the model using `model.fit(X_train, Y_train)`.
11. **Classification Prediction:** Predicted salary categories using `model.predict()`.
12. **Probability Prediction:** Calculated class probabilities using `model.predict_proba()`.
13. **Model Evaluation:** Evaluated the model using Accuracy, Precision, Recall and F1 Score.
14. **Confusion Matrix:** Generated and visualized the confusion matrix.
15. **Classification Report:** Generated a detailed classification report.
16. **Probability Visualization:** Plotted the Logistic Regression probability curve using `TotalWorkingYears`.
17. **New Data Prediction:** Predicted salary categories for employees with different working experience values.
18. **Model Serialization:** Saved the trained Logistic Regression model and scaler using Pickle.

---

## 📈 5. Model Performance Results

The Logistic Regression model is evaluated using the testing dataset.

The dataset is divided into training and testing sets using an 80:20 ratio.

### Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **Accuracy** | Measures the percentage of correct predictions |
| **Precision** | Measures how many predicted positive cases are correct |
| **Recall** | Measures how many actual positive cases are correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |

The actual evaluation values are generated when the Google Colab notebook is executed.

### Confusion Matrix

The confusion matrix is used to evaluate the classification results.

It displays:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

The confusion matrix and classification report are generated using the testing data.

### Key Result

The model classifies employees into Low Salary and High Salary categories based on their total working years.

The final accuracy and other evaluation metrics depend on the actual model execution.

---

## 🔍 6. Prediction on New Data

The trained Logistic Regression model can be tested with new employee working experience values.

For example:

| Parameter | Value |
| :--- | :--- |
| Total Working Years | 5 |
| Total Working Years | 7 |

The model predicts whether the employee belongs to the Low Salary or High Salary category.

The predicted category and probability values are generated using the trained model.

---

## 🛠️ 7. Technologies Used

| Technology | Purpose |
| :--- | :--- |
| Python 3.x | Programming language |
| NumPy | Numerical calculations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
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
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

### Step 3: Dataset Setup

Download the following CSV dataset:

`HR-Employee-Attrition.csv`

Place the dataset in your project directory or upload it directly to Google Colab.

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook Logistic_Regrassion.ipynb
```

Alternatively, open the notebook using Google Colab and execute each code cell.

---

## 📁 9. Project Structure

```text
Logistic_Regrassion-AD3365/
│
├── Logistic_Regrassion.ipynb
│
├── HR-Employee-Attrition.csv
│
├── logistic_model.pkl
│
├── scaler.pkl
│
└── README.md
```

### File Description

- `Logistic_Regrassion.ipynb` – Complete implementation of the Logistic Regression model.
- `HR-Employee-Attrition.csv` – IBM HR employee dataset used for training and testing.
- `logistic_model.pkl` – Saved trained Logistic Regression model.
- `scaler.pkl` – Saved StandardScaler object.
- `README.md` – Project documentation.

**Note:** The Pickle files are generated by executing the model-saving section of the notebook.

---

## 🎯 10. Learning Outcomes

Through this project, I gained practical knowledge of:

- Fundamentals of Supervised Machine Learning.
- Understanding Logistic Regression.
- Difference between Linear Regression and Logistic Regression.
- Loading and exploring CSV datasets.
- Converting numerical salary values into binary classes.
- Data preprocessing and feature selection.
- Feature scaling using StandardScaler.
- Training and testing classification models.
- Understanding the Sigmoid Function.
- Predicting class probabilities.
- Evaluating models using Accuracy, Precision, Recall and F1 Score.
- Understanding and interpreting a Confusion Matrix.
- Saving and loading trained models using Pickle.

---

## 📝 11. Conclusion

This project demonstrates the implementation of Logistic Regression for employee salary classification using the IBM HR Analytics Employee Attrition Dataset.

The model uses `TotalWorkingYears` as the input feature and `SalaryCategory` as the output variable. The salary categories are generated from the median monthly income.

The project provides practical experience in data preprocessing, feature scaling, classification, probability prediction, model evaluation and model serialization using Python and Scikit-Learn.

---

## 👩‍💻 12. GitHub Submission

**Faculty GitHub ID:** [complexiota](https://github.com/complexiota)

**Student GitHub Repository:** [Logistic Regression – AD3365](https://github.com/Gujardg/Logistic_Regrassion-AD3365)

| Project Details | Information |
| :--- | :--- |
| Project Name | Employee Salary Classification |
| Algorithm | Logistic Regression |
| Dataset | IBM HR Analytics Employee Attrition |
| Domain | Machine Learning |
| Programming Language | Python |
| Repository Visibility | Public |

---

## 📄 13. License

This project is created for educational and academic purposes.
