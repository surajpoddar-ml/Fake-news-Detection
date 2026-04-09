
---

```markdown
# 📰 Fake News Detection - SURAJ AI HORIZON

An interactive web application built with **Streamlit** that uses Machine Learning to classify news articles as **REAL** or **FAKE**. This project implements Natural Language Processing (NLP) techniques to analyze text patterns and provide instant predictions.

## 🚀 Features

- **Multiple Vectorization Options:** Choose between `TF-IDF` and `Bag of Words` (CountVectorizer).
- **Interchangeable Classifiers:** Switch between `Linear SVM` and `Naive Bayes` via the sidebar.
- **Dynamic Training:** The model trains and caches based on your selected parameters for optimal performance.
- **Modern UI:** A clean, user-friendly interface with custom CSS for a professional feel.
- **Error Handling:** Robust data validation for CSV inputs and empty text submissions.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **NLP:** TfidfVectorizer, CountVectorizer

## 📋 Prerequisites

Ensure you have the following installed:
- Python 3.8+
- The `fake_or_real_news.csv` dataset in the root directory.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/fake-news-detection.git](https://github.com/your-username/fake-news-detection.git)
   cd fake-news-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit numpy pandas scikit-learn
   ```

3. **Prepare the dataset:**
   Place your `fake_or_real_news.csv` file in the main folder. The CSV must contain at least two columns: `label` (REAL/FAKE) and `text`.

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 🧠 How It Works

1. **Data Loading:** The app loads the dataset and maps labels (`REAL` → 0, `FAKE` → 1).
2. **Feature Extraction:** Text is converted into numerical data using the selected vectorizer (TF-IDF or Bag of Words).
3. **Model Training:** A classifier (SVM or Naive Bayes) is trained on the vectorized text.
4. **Prediction:** When a user inputs an article, the app transforms the text using the *same* fitted vectorizer and predicts the outcome using the trained model.

## 🖥️ Usage

1. Select your preferred **Vectorizer** and **Classifier** from the sidebar.
2. Paste the text of a news article into the text area.
3. Click the **"Check News Validity"** button.
4. The app will display a 🚩 **FAKE** or ✅ **REAL** result based on the analysis.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed by **SURAJ PODDAR***
