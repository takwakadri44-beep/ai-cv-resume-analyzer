import re
import numpy as np
import streamlit as st
import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI CV / Resume Analyzer")
st.caption("AI-powered CV ↔ Job Description matching — no predefined skills dictionary.")

@st.cache_resource(show_spinner="🤖 Loading AI model (first time only)...")
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_pdf(file):
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def tfidf_score(cv, job):
    v = TfidfVectorizer(ngram_range=(1, 2))
    x = v.fit_transform([cv, job])
    return float(cosine_similarity(x[0:1], x[1:2])[0, 0] * 100)

def dynamic_concepts(text, n=15):
    """
    No fixed skills dictionary.
    Concepts are discovered directly from the supplied document
    using TF-IDF n-grams.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        sentences = [text]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=500,
        min_df=1
    )

    matrix = vectorizer.fit_transform(sentences)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda x: x[1],
        reverse=True
    )

    concepts = []
    for term, score in ranked:
        term = term.strip()
        if len(term) < 3:
            continue
        if term not in concepts:
            concepts.append(term)
        if len(concepts) >= n:
            break

    return concepts

def semantic_concept_match(cv_concepts, job_concepts, model):
    if not cv_concepts or not job_concepts:
        return [], [], job_concepts

    cv_emb = model.encode(
        cv_concepts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    job_emb = model.encode(
        job_concepts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    sims = np.matmul(job_emb, cv_emb.T)

    strong, partial, missing = [], [], []

    for i, concept in enumerate(job_concepts):
        best = float(np.max(sims[i]))

        if best >= 0.70:
            strong.append((concept, best))
        elif best >= 0.48:
            partial.append((concept, best))
        else:
            missing.append((concept, best))

    return strong, partial, missing

cv_file = st.file_uploader(
    "📄 Upload CV / Resume",
    type=["pdf"]
)

job_file = st.file_uploader(
    "💼 Upload Job Description",
    type=["txt"]
)

if cv_file and job_file:

    cv_text = clean(extract_pdf(cv_file))
    job_text = clean(
        job_file.read().decode("utf-8", errors="ignore")
    )

    if not cv_text:
        st.error("Could not extract text from the CV PDF.")
        st.stop()

    if not job_text:
        st.error("Job Description is empty.")
        st.stop()

    st.success("Documents loaded successfully.")

    progress = st.progress(0, text="Starting analysis...")

    # 1 — TF-IDF
    progress.progress(20, text="📊 Approach A: TF-IDF + cosine similarity...")
    score_a = tfidf_score(cv_text, job_text)

    # 2 — dynamic concept discovery
    progress.progress(40, text="🔎 Discovering concepts from the documents...")
    cv_concepts = dynamic_concepts(cv_text)
    job_concepts = dynamic_concepts(job_text)

    # 3 — AI model
    progress.progress(55, text="🤖 Loading Sentence Transformer...")
    model = load_model()

    # 4 — semantic similarity
    progress.progress(70, text="🧠 Approach B: semantic similarity...")
    score_b = 0

    embeddings = model.encode(
        [cv_text, job_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    score_b = float(np.dot(embeddings[0], embeddings[1]) * 100)

    # 5 — concept matching
    progress.progress(90, text="🔬 Comparing discovered concepts...")
    strong, partial, missing = semantic_concept_match(
        cv_concepts,
        job_concepts,
        model
    )

    progress.progress(100, text="✅ Analysis complete!")

    st.divider()
    st.header("📊 Matching Results")

    a, b, c = st.columns(3)

    with a:
        st.metric("Approach A — TF-IDF", f"{score_a:.1f}%")

    with b:
        st.metric(
            "Approach B — Sentence Transformers",
            f"{score_b:.1f}%"
        )

    with c:
        winner = (
            "Sentence Transformers"
            if score_b >= score_a
            else "TF-IDF"
        )
        st.metric("Better approach", winner)

    st.divider()
    st.header("🧠 Dynamic Concept Analysis")
    st.caption(
        "No predefined skills list is used. Concepts are discovered "
        "from the uploaded documents and compared semantically."
    )

    x, y, z = st.columns(3)

    with x:
        st.subheader("🟢 Strong")
        for concept, score in strong:
            st.write(f"✓ **{concept}**")

    with y:
        st.subheader("🟡 Partial")
        for concept, score in partial:
            st.write(f"~ **{concept}**")

    with z:
        st.subheader("🔴 Missing / weak")
        for concept, score in missing:
            st.write(f"✗ **{concept}**")

    with st.expander("🔎 Concepts discovered in CV"):
        st.write(", ".join(cv_concepts))

    with st.expander("🔎 Concepts discovered in Job Description"):
        st.write(", ".join(job_concepts))

else:
    st.info("Upload a CV PDF and a Job Description TXT to start.")
