# IPO Success Prediction – Machine Learning Project

### Team 24 — Machine Learning NITKKR 2025

This repository contains the Google Colab notebook and resources used for predicting IPO success using machine‑learning techniques. The project analyzes Indian IPOs (2010–2025) using regression, classification, clustering, and neural networks to forecast:

- Listing Gains
- Long-Term Returns
- Overall IPO Success Category

---

## Project Structure

```
Team-24/
│
├── .gitignore
├── Initial Public Offering.xlsx        # Dataset used for ML model
├── ipo_code.ipynb                      # Google Colab notebook
├── ml_project_final_proposal.pdf       # Initial project proposal
├── report.pdf                          # Final report
└── README.md                           # Project documentation
```

---

## How to Run the Notebook

### Step 1 — Open in Google Colab
1. Upload `ipo_code.ipynb` to Google Colab.  
2. Upload the dataset `Initial Public Offering.xlsx`.  
   - Go to **Files → Upload**

---

### Step 2 — Update Dataset Path

```python
df = pd.read_excel('/content/Initial Public Offering.xlsx')
```

(Use `.csv` if your dataset is CSV.)

---

### Step 3 — Run All Cells

Use **Runtime → Run all** to execute preprocessing, modeling, evaluation, and visualizations.

---

## Short Description

This project predicts IPO outcomes using:

- Regression Models  
- Classification Models  
- Neural Network (MLPClassifier)  
- K-Means Clustering  

The notebook includes insights, graphs, model evaluations, and comparisons.  
The final report summarizes analysis and findings.

---

## Team Members

| Name            | Roll No.    |
|----------------|--------------|
| Inshika Gupta  | 524110027    |
| Ananya         | 524410021    |
| Rukmini Kumari | 524110062    |
