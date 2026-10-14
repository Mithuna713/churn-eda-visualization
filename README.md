# Customer Churn: Exploratory Data Analysis, Visualization and Deep Learning

Case study for **U21ADP05 – Exploratory Data Analysis and Visualization** (Assignment II).
The project explores a real-world telecom dataset, visualizes why customers leave, and trains a
Multi-Layer Perceptron (MLP) to predict churn.

## Dataset
**IBM Telco Customer Churn** – 7,043 customers, 21 fields (demographics, services, contract, billing, churn label).
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Direct CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The script downloads the CSV automatically. If you already have it, save it as `telco.csv` in the project root.

## Objectives
- Understand the structure, quality and statistics of a real-world dataset.
- Clean and preprocess the data (missing values, duplicates, outliers, encoding, scaling, train/validation/test split).
- Create meaningful visualizations covering the seven stages of data visualization.
- Perform univariate, bivariate and multivariate analysis and study discrete and continuous probability distributions.
- Build and evaluate a deep learning model (MLP) with hyperparameter tuning and feature importance.
- Draw conclusions and retention recommendations.

## Methodology
Follows the 7 stages of visualization: **Acquire → Parse → Filter → Mine → Represent → Refine → Interact.**
1. Cleaned `TotalCharges` (11 blanks belong to new customers with tenure 0), checked duplicates and outliers, merged redundant category levels.
2. Encoded categorical fields (24 input features, including engineered `n_addons`), standardized numerical fields, stratified 70:15:15 split.
3. Produced 15 visualizations plus one interactive Plotly chart; ran chi-square, Mann–Whitney, Binomial and continuous-distribution fits.
4. Trained an MLP (Keras/TensorFlow) with dropout, class weights, early stopping and a 12-combination grid search; compared with a logistic regression baseline; computed permutation feature importance.

## Key Findings
- 26.5% of customers churned.
- Churn is highest for short-tenure customers (47.4% in the first year vs 9.5% after four years), month-to-month contracts (42.7% vs 2.8% for two-year), fiber optic users (41.9%) and electronic check payers (45.3%).
- The number of add-on services is over-dispersed compared with a Binomial model; TotalCharges is right-skewed and not normal.
- MLP test results: accuracy 0.747, recall 0.786, F1 0.623, **AUC 0.842** – practically equal to logistic regression (AUC 0.845).

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── src/
│   └── churn_eda_dl.py          # full pipeline: EDA, visualization, model, evaluation
├── report/
│   ├── Assignment_II_Case_Study_Report.docx
│   ├── Assignment_II_Case_Study_Report.pdf
│   └── IEEE_Research_Paper.docx
├── figs/                         # all generated charts (15 PNG files)
└── outputs/
    ├── results.json              # numerical results used in the report
    └── interactive_tenure_vs_charges.html
```

## How to Run
```bash
pip install -r requirements.txt
python src/churn_eda_dl.py
```
Charts are saved to `figs/`, numerical results to `results.json`, and the interactive chart to
`interactive_tenure_vs_charges.html` (in the folder you run from). Training 12 models takes a few minutes on CPU.

## Tools
Python, pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly, scikit-learn, TensorFlow/Keras.

## Author
Mithuna R – Department of Computer Science and Engineering, KPR Institute of Engineering and Technology, Coimbatore.
