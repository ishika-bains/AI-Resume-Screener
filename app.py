import streamlit as st
import pandas as pd
import re
import spacy
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# Load NLP model
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("UpdatedResumeDataSet.csv")
    except:
        return None

df_pool = load_dataset()
# --------------------------------------
nlp = load_nlp()

# Cleaning logic from your notebook
def clean_text(text):
    text = text.lower()
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    text = ""
    try:
        # Use 'strict=False' to allow PyPDF2 to fix minor formatting issues automatically
        reader = PdfReader(file, strict=False) 
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Skill Gap Analysis function
def get_missing_keywords(jd_text, resume_text):
    jd_doc = nlp(jd_text.lower())
    resume_doc = nlp(resume_text.lower())
    jd_keywords = set([token.text for token in jd_doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])
    resume_keywords = set([token.text for token in resume_doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])
    missing = jd_keywords - resume_keywords
    garbage = {'experience', 'team', 'candidate', 'working', 'knowledge', 'years', 'requirements', 'skills', 'role'}
    return [word for word in missing if word not in garbage]

# Dashboard UI
st.title("🎯 AI Resume Screener & Skill Matcher")
st.markdown("Rank resumes against job descriptions using TF-IDF and NLP.")

# Sidebar for Job Description
st.sidebar.header("Job Settings")
jd_input = st.sidebar.text_area("Paste Job Description here:", height=300)

# File Uploader
uploaded_files = st.file_uploader("Upload Files", type=["pdf", "csv"], accept_multiple_files=True)

if jd_input and uploaded_files:
    with st.spinner("Analyzing resumes..."):
        # Process JD
        cleaned_jd = clean_text(jd_input)
        
        # Process uploaded resumes
        resume_data = []
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            cleaned_resume = clean_text(raw_text)
            resume_data.append({"FileName": file.name, "RawText": raw_text, "Cleaned": cleaned_resume})
        
        # Vectorization & Similarity
        all_text = [cleaned_jd] + [r['Cleaned'] for r in resume_data]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Build Results Table
        results_df = pd.DataFrame({
            "Resume Name": [r['FileName'] for r in resume_data],
            "Match Score (%)": [round(s * 100, 2) for s in similarities]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Display Top Results
        st.subheader("✅ Top Matches")
        st.dataframe(results_df, use_container_width=True)

        # Detailed Analysis for Top Candidate
        top_res_name = results_df.iloc[0]["Resume Name"]
        top_res_text = next(item for item in resume_data if item["FileName"] == top_res_name)["RawText"]
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"💡 Skill Gap: {top_res_name}")
            missing = get_missing_keywords(jd_input, top_res_text)
            if missing:
                for skill in missing[:10]:
                    st.write(f"❌ Missing: **{skill}**")
            else:
                st.success("Perfect Match! All skills found.")

        with col2:
    st.subheader("📊 Talent Pool Analysis")
    if df_pool is not None:
        fig, ax = plt.subplots()
        # This creates a bar chart of the top categories in your dataset
        sns.countplot(y="Category", data=df_pool, 
                      order=df_pool['Category'].value_counts().index[:10], 
                      palette="viridis", ax=ax)
        ax.set_title("Top 10 Resume Categories in Database")
        st.pyplot(fig)
    else:
        st.warning("Upload 'UpdatedResumeDataSet.csv' to GitHub to see pool analytics.")

else:
    st.info("Please enter a Job Description and upload at least one resume to start.")
