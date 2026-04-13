# AI Resume Screener & Skill Matcher 🎯

An intelligent recruitment tool that ranks resumes against job descriptions using **Natural Language Processing (NLP)** and **Machine Learning**.

## 🚀 Features
- **Automated Ranking**: Uses TF-IDF and Cosine Similarity to rank resumes based on relevance.
- **Skill Gap Analysis**: Identifies missing technical keywords using the `spaCy` NLP library.
- **Visual Analytics**: Interactive bar charts and distribution plots using Seaborn.
- **Multi-Resume Support**: Upload multiple PDFs simultaneously for bulk screening.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **NLP**: Spacy, TF-IDF (Scikit-learn)
- **Data Handling**: Pandas, NumPy
- **Visuals**: Matplotlib, Seaborn

## 📋 How to Use
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Upload your job description and resume PDFs to see the results.

## 📊 Dataset
The project uses the `UpdatedResumeDataSet.csv` containing categorized resume data for training and context.
