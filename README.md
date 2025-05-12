
# 💼 Job Offer Scam Detection

This project addresses the growing threat of job scam emails by developing a machine learning-based email classification system. Scammers use increasingly sophisticated tactics to impersonate legitimate employers, posing serious risks such as identity theft, financial loss, and emotional distress.

By leveraging natural language processing, feature engineering, and multiple machine learning classifiers, this tool helps job seekers identify fraudulent job offers in real time and supports broader goals in cybersecurity and digital trust.

---

## 🛑 Problem Statement

Job scam emails often mimic real offers, making them hard to detect using traditional spam filters. This project aims to bridge that gap by:

- Creating a hybrid model trained on real, reported, and synthetic scam email samples.
- Using text, metadata, and behavioral features for classification.
- Improving detection precision without wrongly flagging legitimate offers.

---

## ✨ Features

- Hybrid feature extraction: text, sender behavior, urgency terms, scam keywords, etc.
- Custom metadata indicators: non-HTTPS links, bait phrases, identity info requests.
- Balanced dataset with real-world diversity (Enron, surveys, synthetic).
- SHAP-based model interpretability for feature importance.
- Pre-trained Random Forest model (`grid_rf_model.pkl`) with high precision and recall.

---

## 🗂 Project Structure

```
├── complete_model_workflow/
│   └── ipynb/
│       ├── ML_Group3_EDA.ipynb
│       ├── ML_Group3_ModelTraining.ipynb
│       └── ML_Group3_FinalModel.ipynb
├── *.ipynb (Model comparison and individual experiments)
├── email_job_scam_cleaned.csv
├── grid_rf_model.pkl
├── README.md
```

---

## 📊 Dataset

- **email_job_scam_cleaned.csv** – Preprocessed and labeled dataset.
- Sources:
  - Original Job Postings dataset
  - Enron Email dataset
  - Survey-collected scam samples
  - Synthetically generated scam emails
- Balanced via class weighting and oversampling for fair model training.

---

## 🤖 Models Implemented

- Random Forest (🏆 Best Performer – Accuracy: 96.7%)
- Decision Tree
- Logistic Regression
- Naive Bayes
- SVM (Linear, RBF, Polynomial)
- K-Nearest Neighbors
- AdaBoost & XGBoost

Hyperparameter tuning: GridSearchCV  
Model evaluation metrics: Accuracy, F1, Precision, Recall, MCC, ROC-AUC

---

## 🧰 Tech Stack

- **Languages**: Python 3.10
- **Frameworks & Libraries**:  
  `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`
- **Tools**:  
  Jupyter Notebook, Google Colab, VS Code, Git, GitHub, Trello

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Explore:
   - `complete_model_workflow/ipynb` → Full pipeline (EDA to evaluation)
   - `grid_rf_model.pkl` → Load and test best model
3. Predict on new data by importing the pickled model:
   ```python
   import joblib
   model = joblib.load("grid_rf_model.pkl")
   ```

---

## 📈 Results

| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Random Forest     | 96.7%    | 81.3%     | 66.8%  | 73.4%    |
| SVM (RBF Kernel)  | 96.4%    | 89.1%     | 53.7%  | 67.0%    |
| Naive Bayes       | 92.7%    | 47.3%     | 63.0%  | 54.0%    |

SHAP analysis confirmed key scam indicators:  
`scam_keywords`, `sender_domain_reputation`, `non_https_links`, `id_info_requested`, and `urgency_terms`.

---

## 🌍 Sustainability Impact

Aligned with [UN Sustainable Development Goals (SDG)](https://sdgs.un.org/goals):

- **SDG 8** – Decent Work and Economic Growth  
- **SDG 9** – Industry, Innovation, and Infrastructure  
- **SDG 16** – Peace, Justice, and Strong Institutions  

Prevents economic and emotional exploitation of vulnerable job seekers.

---

## 🔮 Future Work

- Integration with email clients as a browser plugin (Gmail, Outlook)
- Real-time stream processing for inboxes
- Transformer-based model exploration (BERT, RoBERTa)
- Multilingual scam detection support
- Extension to rental scams, fake scholarships, and more

---

## 🤝 Contributors

- **Sreenidhi Hayagreevan** – Data Syntesis, Random Forest modeling, documentation  
- **Shivani Beri** – Feature engineering, SVM & KNN models  
- **Janani Kripa Manoharan** – EDA, Naive Bayes modeling, visualizations  
- **Laxmi Thrishitha Kalvakota** – Boosting models, benchmarking, project coordination


---

## 🙏 Acknowledgments

This project was developed as part of the DATA245 – Machine Learning course at San Jose State University.

We would like to sincerely thank:

- **Dr. Vishnu Pendyala**, our course instructor, for his expert guidance, encouragement, and constructive feedback throughout the semester.
- The **Department of Applied Data Science**, for providing the infrastructure, resources, and academic support needed to carry out this project effectively.
- All our **LinkedIn and peer network contributors**, who helped us gather real-world scam email patterns for enhanced model realism.
- Tools such as **Google Colab**, **SciSpace**, and **ChatGPT**, which played a role in experimentation, collaboration, and report enhancement.

This project reflects a genuine attempt to turn a personal negative experience with online job scams into a positive, impactful solution for others.
