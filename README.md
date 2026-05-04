Data Analyrics Practice

A collection of end-to-end data science projects covering the full workflow: from data cleaning and exploratory data analysis (EDA) to building predictive models with supervised and unsupervised learning techniques.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |

---

## Project Structure

### 1. IT Salary Prediction — `main_example.py`
> EDA of the IT labor market and specialist income prediction.

- **Work done:** Dataset cleaning, data aggregation via `pivot_table`, and income distribution visualization
- **Models & Methods:** Linear Regression (Features: year, experience level, remote ratio, company size)

---

### 2. Sales Forecasting — `ice_cream_sales_prediction.py`
> Predicting sales volume based on temperature and time of day.

- **Work done:** Feature Engineering (datetime parsing, converting time into numerical format)
- **Models & Methods:** Comparison between Linear and Polynomial Regression (Evaluation: R-squared)

---

### 3. Medical Binary Classification — `medical_rf_classification.py`
> Predicting disease probability based on physiological metrics.

- **Work done:** Missing value imputation (mean/median), feature scaling (`StandardScaler`), and correlation matrix visualization
- **Models & Methods:** Random Forest Classifier, Feature Importance extraction

---

### 4. Netflix Content Classification & Clustering — `netflix_ml_tasks.py`
> Categorizing content formats and identifying hidden genre-based groups.

- **Work done:** Numerical data extraction from strings via RegEx, categorical encoding (`LabelEncoder`, `get_dummies`)
- **Models & Methods:**
  - **Supervised:** Decision Tree — Movie vs. TV Show classification
  - **Unsupervised:** K-Means — Content clustering into 5 genre groups

---

## Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Prepare data:**

Place all `.csv` dataset files inside the `data/` directory before running any scripts.

---

## Repository Structure

```
├── data/                          # Dataset files (.csv)
├── ds_salaries_analysis.py        # IT Salary Prediction
├── ice_cream_sales_prediction.py  # Sales Forecasting
├── medical_rf_classification.py   # Medical Classification
├── netflix_ml_tasks.py            # Netflix Content Analysis
└── README.md
```
