# 📉 Customer Churn Prediction

## TL;DR
Built an **end-to-end customer churn prediction system** using LightGBM, calibrated a **business-driven decision threshold (0.35)** based on precision–recall trade-offs, and deployed the final pipeline via a **Streamlit application** to support customer retention targeting.

---

## 📌 Problem Statement
Customer churn is a major challenge for subscription-based businesses. Accurately identifying customers who are likely to churn enables organizations to take proactive retention actions, reduce revenue loss, and improve customer lifetime value.

This project develops a **production-style machine learning system** that predicts the probability of customer churn and converts model outputs into **actionable business decisions**.

---

## Dataset
- **Source:** Public Telco Customer Churn dataset  
- **Size:** ~7,000 customers  
- **Target Variable:** `Churn` (Yes / No)  
- **Class Imbalance:** ~26% churn rate  

The dataset includes a mix of **categorical** and **numerical** features related to:
- Customer tenure
- Contract type
- Billing and payment methods
- Charges and usage behavior

---

##  Exploratory Data Analysis (EDA)
Key insights from EDA include:
- Customers on **month-to-month contracts** churn significantly more than those on long-term contracts
- **Electronic check** payment method is strongly associated with higher churn
- Customers with **short tenure** are much more likely to churn
- Long-tenure customers exhibit strong retention behavior

EDA is documented in `notebooks/01_eda.ipynb`.

---

## Modeling Approach

### Models Evaluated
- Logistic Regression (baseline)
- LightGBM (final model)

### Why LightGBM?
- Effectively captures **non-linear relationships** in categorical features
- Handles feature interactions efficiently
- Provides better recall–precision trade-offs for churn prediction compared to the baseline model

---

##  Model Performance

| Model               | ROC-AUC |
|--------------------|---------|
| Logistic Regression | ~0.84   |
| LightGBM           | ~0.83   |

Although ROC-AUC scores are similar, **LightGBM performs better when evaluated using business-relevant thresholds**.

---

## Business Threshold Selection
Rather than using a default probability threshold (0.5), a **precision–recall analysis** was conducted to select an optimal decision threshold.

- **Selected Threshold:** **0.35**
- **Rationale:**  
  This value lies at the *knee* of the precision–recall curve, maintaining high recall (~82%) while achieving acceptable precision, balancing retention cost with churn capture.

### Decision Rule
> Customers with predicted churn probability ≥ **0.35** are flagged for retention outreach.

This transforms model predictions into a **clear operational policy**.

---

## Streamlit Application
A Streamlit application was built to make the model accessible to non-technical stakeholders.

### Features
- Interactive input form for customer attributes
- Displays:
  - Churn probability
  - Risk classification (High / Low)
- Applies the calibrated business threshold (0.35)

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py

---

## Project Structure
customer-churn-prediction/
├── app/
│   └── app.py                 # Streamlit application
├── models/
│   └── churn_lightgbm_pipeline.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_model.ipynb
├── data/
├── requirements.txt
├── README.md
└── .gitignore

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Matplotlib, Seaborn
- Streamlit
- Git & GitHub

## Key Takeaways
- Built a reproducible ML pipeline with preprocessing and modeling
- Tuned decision thresholds based on business impact, not defaults
- Deployed a production-style Streamlit app for decision support
- Demonstrated end-to-end applied machine learning skills

## Future Improvements
- Cost-sensitive threshold optimization
- Retention ROI simulation
- Model calibration (Platt / Isotonic)
- Cloud deployment (Streamlit Cloud or Docker)

## ✍️ Author

Prince Munene
Aspiring Data Scientist | Machine Learning & Analytics
GitHub: https://github.com/PrinceMunene-code