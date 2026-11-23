# Dubai Rental Property Analysis (CDS 2413)

Group project for **CDS 2413 Programming for Data Analytics**, Semester 202610.

## Group Members
- Haya
- Shaikha
- **Course:** CDS 2413 Programming for Data Analytics, Higher Colleges of Technology

## Project Overview
This project analyses 25,000 Dubai rental property listings to understand how property characteristics relate to annual rent, and builds machine learning models to predict and group rental prices.

- **Dataset:** `rentals.csv` (25,000 rows, 24 columns)
- **Dependent variable:** `annual_rent_usd`
- **Main predictors:** area_sqft, bedrooms, parking_spaces, metro_distance_min, property_category, property_type, furnishing, community
- **Language:** Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn)

## Repository Structure
```
CDS2413-Dubai-Rental-Analysis/
|-- data/
|   |-- rentals.csv                 Dubai rental dataset used in all deliverables
|-- code/
|   |-- deliverable1.py             D1: descriptive statistics, sampling, charts, correlation, hypothesis testing
|   |-- deliverable2.py             D2: regression, classification, clustering
|-- reports/
|   |-- CDS2413_Deliverable1_Data_Analysis_Report.docx
|   |-- CDS2413_Deliverable2_ML_Model_Report.docx
|-- requirements.txt                Python libraries needed
|-- .gitignore
|-- README.md
```

## Deliverables
| Deliverable | CLO | Content |
|---|---|---|
| D1 Data Analysis Report | CLO1, CLO2 | Purpose, variables, descriptive statistics, random and systematic sampling, charts, correlation, hypothesis testing |
| D2 Machine Learning Model Report | CLO3 | Simple and multiple linear regression, Logistic Regression, KNN, Naive Bayes, Decision Tree, K-Means and Hierarchical clustering |
| D3 Data Versioning | CLO4 | This GitHub repository and its commit history |

## Key Results
- Multiple Linear Regression: R² = 0.8160 on the test set
- Best classifier: Logistic Regression, accuracy 94.18%
- K-Means and Hierarchical Clustering each identified three property groups

## How to Run
Run the scripts from the main project folder so the data path works:
```
pip install -r requirements.txt
python code/deliverable1.py
python code/deliverable2.py
```
