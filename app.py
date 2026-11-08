import re
import numpy as np
import streamlit as st
import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI CV / Resume Analyzer")
st.write(
    "Compare any CV with any Job Description using AI — "
    "without a predefined skills dictionary."
)


@st.cache_resource
def load_models():

    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    keyword_model = KeyBERT(model=model)

    return model, keyword_model


def extract_pdf(file):

    data = file.read()

    document = fitz.open(
        stream=data,
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text()

    return text


def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tfidf_similarity(cv, job):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        [cv, job]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return similarity * 100


def semantic_similarity(cv, job, model):

    embeddings = model.encode(
        [cv, job],
        normalize_embeddings=True
    )

    similarity = np.dot(
        embeddings[0],
        embeddings[1]
    )

    return similarity * 100


def extract_concepts(text, keyword_model):

    keywords = keyword_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=20,
        use_mmr=True,
        diversity=0.6
    )

    return [
        keyword
        for keyword, score in keywords
    ]


def compare_concepts(
    cv_concepts,
    job_concepts,
    model
):

    if not cv_concepts:

        return [], [], job_concepts

    cv_embeddings = model.encode(
        cv_concepts,
        normalize_embeddings=True
    )

    job_embeddings = model.encode(
        job_concepts,
        normalize_embeddings=True
    )

    similarities = np.matmul(
        job_embeddings,
        cv_embeddings.T
    )

    strong = []
    partial = []
    missing = []

    for i, concept in enumerate(job_concepts):

        best_similarity = np.max(
            similarities[i]
        )

        if best_similarity >= 0.72:

            strong.append(concept)

        elif best_similarity >= 0.48:

            partial.append(concept)

        else:

            missing.append(concept)

    return strong, partial, missing


st.sidebar.title("⚙️ Settings")

st.sidebar.info(
    """
This application does NOT use a fixed
skills dictionary.

The AI discovers important concepts
directly from the CV and Job Description.
"""
)


col1, col2 = st.columns(2)


with col1:

    cv_file = st.file_uploader(
        "📄 Upload CV / Resume",
        type=["pdf"]
    )


with col2:

    job_file = st.file_uploader(
        "💼 Upload Job Description",
        type=["txt"]
    )


if cv_file and job_file:

    cv_text = clean_text(
        extract_pdf(cv_file)
    )

    job_text = clean_text(
        job_file.read().decode(
            "utf-8",
            errors="ignore"
        )
    )

    if not cv_text:

        st.error(
            "Could not extract text from the CV."
        )

        st.stop()

    if not job_text:

        st.error(
            "Job Description is empty."
        )

        st.stop()


    st.success(
        "Documents loaded successfully!"
    )


    with st.spinner(
        "🤖 AI is analyzing the documents..."
    ):

        model, keyword_model = load_models()

        tfidf_score = tfidf_similarity(
            cv_text,
            job_text
        )

        semantic_score = semantic_similarity(
            cv_text,
            job_text,
            model
        )

        cv_concepts = extract_concepts(
            cv_text,
            keyword_model
        )

        job_concepts = extract_concepts(
            job_text,
            keyword_model
        )

        strong, partial, missing = compare_concepts(
            cv_concepts,
            job_concepts,
            model
        )


    st.divider()

    st.header("📊 Matching Results")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "TF-IDF",
            f"{tfidf_score:.1f}%"
        )


    with col2:

        st.metric(
            "Sentence Transformers",
            f"{semantic_score:.1f}%"
        )


    with col3:

        if semantic_score >= tfidf_score:

            winner = "Sentence Transformers"

        else:

            winner = "TF-IDF"

        st.metric(
            "Better Approach",
            winner
        )


    st.divider()

    st.header(
        "🧠 Dynamic Skill / Concept Analysis"
    )

    st.write(
        "The concepts below are discovered "
        "automatically from the documents."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader(
            "🟢 Strong Match"
        )

        for concept in strong:

            st.write(
                "✓",
                concept
            )


    with col2:

        st.subheader(
            "🟡 Partial Match"
        )

        for concept in partial:

            st.write(
                "~",
                concept
            )


    with col3:

        st.subheader(
            "🔴 Missing / Weak"
        )

        for concept in missing:

            st.write(
                "✗",
                concept
            )


    st.divider()

    st.subheader(
        "🔎 Concepts discovered in CV"
    )

    st.write(
        ", ".join(cv_concepts)
    )


    st.subheader(
        "🔎 Concepts discovered in Job Description"
    )

    st.write(
        ", ".join(job_concepts)
    )


else:

    st.info(
        "Upload a CV PDF and a Job Description TXT to start."
  )
