# 📈 Predicting Initial Public Offering (IPO) Success Using Machine Learning  
### Machine-Learning-NITKKR-2025 — Team 24

A complete machine-learning project that predicts IPO success in India using regression, classification, clustering, and neural networks.  
The project is implemented fully from scratch, including custom preprocessing, Min–Max scaling, Gradient Descent, Gini Impurity, Bootstrapping, MLP backpropagation, and K-Means.

---

## 📝 1. Project Overview  
IPO performance is uncertain and highly risky for investors. Traditional financial analysis cannot capture complex, non-linear relationships such as issue size, subscription demand, and listing gains.

This project builds a **data-driven framework** using machine learning to:

- Predict short-term **Listing Gains**  
- Predict long-term **Current Gains**  
- Classify IPOs as **Successful vs. Unsuccessful**  
- Segment IPOs into natural groups using **K-Means Clustering**  
- Explore deep patterns using an **MLP Neural Network**

This framework helps make IPO investing more informed and less risky for students and retail investors.

---

## 🎯 2. Problem Statement  
Given detailed IPO data (2010–2025) containing issue size, subscription numbers, offer price, and market performance, the project aims to predict:

1. **Short-term performance** → Listing Gain (Regression)  
2. **Long-term performance** → Current Gains (Regression)  
3. **IPO Success** → Binary Classification  
4. **Market Segments** → Clustering into natural IPO archetypes  

---

## 🧠 3. Features  
- End-to-end ML pipeline (cleaning → preprocessing → modeling → evaluation)  
- Regression models implemented from scratch  
- Custom Decision Tree & Random Forest implementation  
- Custom MLP Neural Network (forward + backward propagation)  
- K-Means clustering with silhouette evaluation  
- Comparative model analysis  
- Visualizations for all tasks  

---

## 🗂 4. Dataset  
**Source:** Kaggle – Indian IPO Dataset (2010–2025)  
**Link:** https://www.kaggle.com/datasets/karanammithul/ipo-data-india-2010-2025  

### Key Features Used  
- Issue Size (crores)  
- Subscription demand: QIB, HNI, RII, Total  
- Offer Price  
- Listing Price → Used to compute *Listing Gain*  
- CMP (Current Market Price) → Used to compute *Current Gains*  
- IPO_Year (engineered from date)  

### Target Variables  
- **Listing Gain** (Regression)  
- **Current Gains** (Regression)  
- **IPO_Success_Binary** (Classification)

---

## 🧹 5. Preprocessing  
- Removed irrelevant columns  
- Handled missing values using **mean imputation**  
- Engineered IPO_Year from Date  
- Created binary success label:  
  - Success = 1 if Listing Gain > 0  
- Applied **scratch Min–Max Scaling**  
- 80% training / 20% test split  

---

## 🔧 6. Methodology

### **6.1 Regression**
Models:
- Linear Regression (Gradient Descent)  
- MLP Regression (ReLU + Linear Output)

Targets:
- Listing Gain  
- Current Gains  

---

### **6.2 Classification**
Models:
- Decision Tree (Gini Impurity)  
- Random Forest (Bootstrapping + Majority Voting)  
- MLP Classifier (Sigmoid Output)

Target:
- IPO Success (1 = Successful, 0 = Unsuccessful)

---

### **6.3 Clustering**
- K-Means (K=3)  
- Evaluated using Silhouette Score  
- Features included: Issue Size, Subscription Total, Listing Gain  

---

## 📊 7. Results

### **7.1 Regression Performance**

| Model | Target | R² | MAE | RMSE |
|-------|--------|------|-----------|-------------|
| Linear Regression | Listing Gain | 0.5658 | 14.11 | 20.20 |
| MLP Regression | Listing Gain | **0.6387** | **11.73** | **18.43** |
| Linear Regression | Current Gains | 0.1981 | 113.61 | 198.55 |
| MLP Regression | Current Gains | −0.0000 | 126.70 | 221.72 |

**Insights:**  
- MLP outperformed Linear Regression for short-term prediction.  
- Both models struggled for long-term predictions → more features needed.

---

### **7.2 Classification Performance**

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|---------|-----------|
| Decision Tree | 0.7345 | 0.7952 | 0.8354 | 0.8148 |
| Random Forest | **0.7876** | **0.8161** | **0.8987** | **0.8554** |
| MLP Classifier | 0.6991 | 0.6991 | **1.00** | 0.8229 |

**Best Model:** **Random Forest**  
**Note:** MLP classifier predicted only class “1”, suggesting imbalance or optimization issues.

---

### **7.3 Clustering Results**
- **Silhouette Score:** 0.3681  
- **K=3 clusters** represented:

| Cluster | Characteristics | Investment Insight |
|---------|------------------------|-----------------------------|
| Cluster 2 | Very high demand, highest listing gain | High-return IPOs |
| Cluster 1 | Large issue size, moderate demand | Large-cap stable IPOs |
| Cluster 0 | Mid-size, moderate demand | Moderate-return IPOs |

---

## 📁 8. Project Structure  
```
Team-24/
├── data/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Models.ipynb
├── src/
│   ├── preprocessing.py
│   ├── regression.py
│   ├── classification.py
│   ├── mlp.py
│   ├── clustering.py
├── results/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 9. Usage

### **Clone the repository**
```bash
git clone https://github.com/Machine-Learning-NITKKR-2025/Team-24.git
cd Team-24
```

### **Install dependencies**
```bash
pip install -r requirements.txt
```

### **Run the main script**
```bash
python main.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook
```

---

## 🛠 10. Technologies Used  
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn (for comparison)  
- Custom scratch implementations  
- Google Colab  

---

## 🔮 11. Future Work  
- Improve MLP classification performance with hyperparameter tuning  
- Include external financial indicators (Sensex, Nifty trends, P/E ratios)  
- Perform feature importance analysis (Random Forest, SHAP)  
- Experiment with advanced algorithms (XGBoost, LSTM for long-term gains)  

---

## 👥 12. Team Members  
- **Inshika Gupta (524110027)** – Preprocessing, Linear Regression, Metrics  
- **Ananya (524410021)** – Classification Models, Visualization  
- **Rukmini Kumari (524110062)** – K-Means Clustering, Comparative Analysis  
- **All Members** – Neural Network (MLP) development  

---

## 📚 13. References  
- Kaggle IPO Dataset (2010–2025)  
- Groww & Upstox platforms  
- Google Colab documentation  
