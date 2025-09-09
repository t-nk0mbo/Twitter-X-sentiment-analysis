# COVID-19 Tweet Sentiment Analysis

This project analyzes the sentiment of tweets related to COVID-19 using natural language processing (NLP) and machine learning techniques. The goal was to clean and preprocess raw tweets, extract meaningful features, and evaluate different models for sentiment classification.

## Case Study Link: 
https://animated-talos-5ae.notion.site/Text-Classification-Sentiment-Analysis-1c21204ea4d6805a8f4efa650e75dd07
---

## 📊 Dataset
- Source: [Corona NLP Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)  
- Columns: `UserName, ScreenName, Location, TweetAt, OriginalTweet, Sentiment`  
- Sentiment classes were consolidated into **3 categories**:
  - **0** → Negative (Extremely Negative, Negative)  
  - **1** → Neutral  
  - **2** → Positive (Positive, Extremely Positive)  

---

## 🔎 Key Steps

1. **Exploratory Data Analysis**
   - Checked label distribution in training and test sets
   - Visualized class imbalance

2. **Preprocessing**
   - Removed unnecessary columns (e.g., `Location`)
   - Cleaned text (lowercasing, removing links, punctuation, and numbers)
   - Created a `Processed_Text` column for modeling
   - Mapped multi-class sentiment labels into 3 categories (Negative, Neutral, Positive)

3. **Feature Engineering**
   - Applied **TF-IDF Vectorization** (max 5,000 features, stopwords removed)
   - Visualized top contributing words for sample tweets

4. **Modeling**
   - **Naïve Bayes (MultinomialNB)**
   - **Random Forest Classifier**
   - **Support Vector Machine (SVM)**
   - Evaluated using precision, recall, and F1-score

---

## ⚙️ Tech Stack
- **Python**: pandas, numpy, re, matplotlib, seaborn  
- **NLP & ML**: scikit-learn (TF-IDF, Naïve Bayes, Random Forest, SVM)  

---

## 📈 Results
- **Naïve Bayes**: Performed well on imbalanced classes, fast and efficient.  
- **Random Forest**: Balanced accuracy across classes, interpretable feature importance.  
- **SVM**: Achieved strong overall accuracy with linear kernel.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/covid19-tweet-sentiment-analysis.git
   cd covid19-tweet-sentiment-analysis
2. Run the notebook or script:
   jupyter notebook "Covid19_Tweet_Sentiment_Analysis.ipynb"
