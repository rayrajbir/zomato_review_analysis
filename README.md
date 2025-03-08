# Zomato Review Analysis

## 📌 Project Overview
This project focuses on **Natural Language Processing (NLP) and Sentiment Analysis** to extract insights from **Zomato user reviews**. The goal is to analyze customer sentiment, detect trends, and gain valuable insights into restaurant reviews.

## 🚀 Features
- **Data Preprocessing** (Cleaning, Tokenization, Lemmatization, Stopword Removal)
- **Sentiment Analysis** (Positive, Negative, Neutral classification)
- **Machine Learning Models** for sentiment prediction
- **Data Visualization** using Matplotlib & Seaborn
- **Deployed using Streamlit (Optional)**

## 📂 Dataset
The dataset consists of **user reviews, ratings, and restaurant details**. You can use:
- **Zomato Kaggle Dataset**: [Link](https://www.kaggle.com/datasets/manakverma/zomato-reviews)

## 🛠️ Technologies Used
- **Python**
- **NLTK & SpaCy** (for NLP preprocessing)
- **VADER & TextBlob** (for sentiment analysis)
- **Scikit-Learn** (for ML models)
- **Pandas & NumPy** (for data handling)
- **Matplotlib & Seaborn** (for data visualization)
- **Flask/Streamlit** (for deployment - optional)

## ⚙️ Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/rayrajbir/zomato_review_analysis.git
   cd zomato_review_analysis
   ```
2. **Create a Virtual Environment** (Recommended)
   ```sh
   python -m venv zom_env
   source zom_env/bin/activate  # (Mac/Linux)
   zom_env\Scripts\activate  # (Windows)
   ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download NLTK Resources** (If not downloaded)
   ```sh
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
5. **Run the Analysis**
   ```sh
   python main.py
   ```

## 📊 Example Visualizations
Here are some insights you can generate:
- **Sentiment Distribution Chart**
- **Word Cloud of Most Common Words**
- **Top Positive & Negative Reviews**

## 🤖 Future Enhancements
- Implement **Deep Learning (LSTMs, Transformers)** for improved sentiment detection
- Build an **interactive dashboard** for review insights
- Deploy as a **web app using Streamlit**

## 📝 License
This project is licensed under the **MIT License**.

## 🙌 Contributions
Feel free to **fork this repo, open issues, or submit pull requests** to improve this project.

## 📬 Contact
For questions or collaboration, reach out via [GitHub Issues](https://github.com/rayrajbir/zomato_review_analysis/issues).
