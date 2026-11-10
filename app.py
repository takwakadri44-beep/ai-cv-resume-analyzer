import os
import re
import json
import time
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------------------------------
# 1. إعدادات الصفحة
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartResume AI - Academic Semantic Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

GEMINI_MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS_LLM = 15000

# ----------------------------------------------------------------------------
# 2. التكوين التلقائي للمفتاح من Secrets بدون إظهار أي خانة
# ----------------------------------------------------------------------------
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        genai.configure(api_key=api_key)
except Exception as e:
    pass

if not api_key and os.environ.get("GEMINI_API_KEY"):
    api_key = os.environ.get("GEMINI_API_KEY").strip()
    genai.configure(api_key=api_key)

# ----------------------------------------------------------------------------
# 3. مخططات Pydantic v2
# ----------------------------------------------------------------------------
class JobRequirementsSchema(BaseModel):
    technical_skills: List[str] = Field(default_factory=list)
    software_tools: List[str] = Field(default_factory=list)
    engineering_competencies: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    education_requirements: List[str] = Field(default_factory=list)
    experience_requirements: List[str] = Field(default_factory=list)
    job_responsibilities: List[str] = Field(default_factory=list)

# ----------------------------------------------------------------------------
# 4. تهيئة نموذج التضمين
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_semantic_model(model_name: str = EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)

with st.spinner("جاري تهيئة محرك الذكاء الاصطناعي الدلالي... ⏳"):
    embed_model = load_semantic_model()

# ----------------------------------------------------------------------------
# 5. تنسيق الواجهة (SaaS Styling)
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #0F172A; color: #F1F5F9; border-right: 1px solid #1E293B; }
    section[data-testid="stSidebar"] .stMarkdown { color: #94A3B8; }
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #2563EB 100%);
        padding: 1.8rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.15);
    }
    .hero-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; color: #FFFFFF; }
    .hero-subtitle { font-size: 0.95rem; color: #93C5FD; margin: 0; }
    .saas-card { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.03); margin-bottom: 1rem; }
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%); color: white; border-radius: 10px;
        font-size: 1.05rem; font-weight: 600; padding: 12px 24px; border: none; width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25); transition: all 0.2s ease;
    }
    .stat-card { padding: 14px 18px; border-radius: 10px; text-align: center; }
    .stat-card-green { background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; }
    .stat-card-amber { background: #FEFCE8; border: 1px solid #FEF08A; color: #854D0E; }
    .stat-card-red   { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
    .stat-number { font-size: 1.8rem; font-weight: 800; line-height: 1.2; }
    .stat-label { font-size: 0.85rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. الشريط الجانبي (نظيف بدون أي خانة مفاتيح)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 SmartResume AI")
    st.markdown("<p style='font-size: 0.8rem; color: #94A3B8;'>نظام التوظيف المؤسسي والأكاديمي</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav_choice = st.radio(
        "قائمة التنقل",
        ["Resume Analysis", "Model Experiment (TF-IDF vs AI)"],
        label_visibility="collapsed"
    )

# ----------------------------------------------------------------------------
# 7. دوال المعالجة والذكاء الاصطناعي
# ----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"خطأ في قراءة ملف الـ PDF: {e}")
        return ""

def extract_requirements_with_gemini(job_description: str, max_retries: int = 3) -> Optional[Dict[str, List[str]]]:
    if not api_key:
        st.error("❌ خطأ: لم يتم العثور على مفتاح Gemini API في إعدادات Secrets. يرجى إضافته في إعدادات التطبيق.")
        return None

    prompt = f"""
Analyze the Job Description below and extract structured requirements into precise, normalized, atomic concepts.
Return your response strictly in valid JSON format matching this schema:
{{
    "technical_skills": ["Skill 1", "Skill 2"],
    "software_tools": ["Tool 1", "Tool 2"],
    "engineering_competencies": ["Competency 1"],
    "soft_skills": ["Soft Skill 1"],
    "education_requirements": ["Education 1"],
    "experience_requirements": ["Experience 1"],
    "job_responsibilities": ["Responsibility 1"]
}}

JOB DESCRIPTION:
{job_description[:MAX_CHARS_LLM]}
"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                text_resp = response.text.strip()
                if "```json" in text_resp:
                    text_resp = text_resp.split("```json")[1].split("```")[0].strip()
                elif "```" in text_resp:
                    text_resp = text_resp.split("```")[1].split("```")[0].strip()
                
                raw_data = json.loads(text_resp)
                validated_data = JobRequirementsSchema(**raw_data)
                return validated_data.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries - 1:
                    st.error(f"فشل التحقق من بنية البيانات: {e}")
                    return None
                time.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"خطأ في الاتصال بـ Gemini API: {e}")
                    return None
                time.sleep(1)
    except Exception as e:
        st.error(f"خطأ في تهيئة نموذج Gemini: {e}")
        return None
    return None

def perform_evidence_based_matching(extracted_reqs: Dict[str, List[str]], cv_text: str, model) -> Dict[str, List[Dict[str, Any]]]:
    cv_segments = [s.strip() for s in re.split(r'\n|[.!?]+', cv_text) if len(s.strip()) > 10]
    if not cv_segments:
        cv_segments = [cv_text]

    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    seg_embs = model.encode(cv_segments, convert_to_tensor=True)

    categories_mapping = {
        "technical_skills": "Technical Skill",
        "software_tools": "Software & Tool",
        "engineering_competencies": "Engineering Competency",
        "soft_skills": "Soft Skill",
        "education_requirements": "Education Requirement",
        "experience_requirements": "Experience Requirement",
        "job_responsibilities": "Job Responsibility"
    }

    results = {}
    for cat_key, cat_label in categories_mapping.items():
        items = extracted_reqs.get(cat_key, [])
        evaluated_items = []
        for item in items:
            item_emb = model.encode(item, convert_to_tensor=True)
            seg_scores = util.cos_sim(seg_embs, item_emb)[:, 0]
            best_idx = seg_scores.argmax().item()
            best_score_val = float(seg_scores[best_idx].item() * 100)
            
            global_score = float(util.cos_sim(cv_emb, item_emb).item() * 100)
            similarity = max(best_score_val, global_score)

            best_evidence = "No supporting evidence found in the CV."
            if seg_scores[best_idx].item() > 0.25:
                best_evidence = cv_segments[best_idx]
            
            has_evidence = best_evidence != "No supporting evidence found in the CV."
            
            if similarity >= 65 and has_evidence:
                classification = "Strong"
                explanation = f"تشابه دلالي عالٍ ({round(similarity, 1)}%)."
            elif (similarity >= 40 and has_evidence) or (similarity >= 55):
                classification = "Partial"
                explanation = f"تشابه دلالي متوسط ({round(similarity, 1)}%)."
            else:
                classification = "Missing"
                explanation = f"تشابه دلالي منخفض ({round(similarity, 1)}%)."

            evaluated_items.append({
                "name": item,
                "category": cat_label,
                "classification": classification,
                "similarity_score": round(similarity, 1),
                "cv_evidence": best_evidence,
                "jd_evidence": item,
                "explanation": explanation
            })
        results[cat_key] = evaluated_items
    return results

# ----------------------------------------------------------------------------
# 8. الصفحة 1: التحليل المهني
# ----------------------------------------------------------------------------
if nav_choice == "Resume Analysis":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">📄 لوحة تحكم تحليل السيرة الذاتية المهنية</div>
            <div class="hero-subtitle">تقييم مدعوم بالذكاء الاصطناعي مع استخراج المهارات بدقة</div>
        </div>
    """, unsafe_allow_html=True)

    col_cv, col_jd = st.columns(2, gap="medium")
    with col_cv:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### 📄 رفع السيرة الذاتية (PDF)")
        uploaded_file = st.file_uploader("رفع ملف PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.caption(f"📎 **{uploaded_file.name}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_jd:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### 💼 الوصف الوظيفي")
        job_description = st.text_area("لصق الوصف الوظيفي", height=100, label_visibility="collapsed", placeholder="ألصق متطلبات الوظيفة هنا...")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✨ بدء التحليل المهني بالذكاء الاصطناعي →"):
        if not api_key:
            st.error("⚠️ مفتاح Gemini API مفقود. تأكد من إضافته في إعدادات Secrets على Streamlit Cloud.")
        elif uploaded_file is not None and job_description.strip() != "":
            with st.spinner("جاري استخراج المفاهيم والمطابقة الدلالية..."):
                cv_text = extract_text_from_pdf(uploaded_file)
                if not cv_text:
                    st.error("تعذر قراءة ملف الـ PDF.")
                else:
                    extracted_reqs = extract_requirements_with_gemini(job_description)
                    if not extracted_reqs:
                        st.error("فشل استخراج المتطلبات عبر Gemini API.")
                    else:
                        matched_results = perform_evidence_based_matching(extracted_reqs, cv_text, embed_model)
                        core_keys = ["technical_skills", "software_tools", "engineering_competencies", "soft_skills"]
                        flat_core = []
                        for k in core_keys:
                            flat_core.extend(matched_results.get(k, []))
                        
                        overall_score = round(sum(i['similarity_score'] for i in flat_core) / len(flat_core), 1) if flat_core else 0.0

                        st.session_state['analyzed_p1'] = True
                        st.session_state['matched_results'] = matched_results
                        st.session_state['overall_match_p1'] = overall_score
                        st.session_state['cv_text'] = cv_text
                        st.session_state['job_description'] = job_description
        else:
            st.warning("⚠️ يرجى رفع ملف الـ PDF وكتابة الوصف الوظيفي.")

    if st.session_state.get('analyzed_p1', False):
        matched_results = st.session_state['matched_results']
        overall_score = st.session_state['overall_match_p1']
        core_keys = ["technical_skills", "software_tools", "engineering_competencies", "soft_skills"]
        skills_flat = []
        for k in core_keys:
            skills_flat.extend(matched_results.get(k, []))

        strong_skills = [s for s in skills_flat if s['classification'] == "Strong"]
        partial_skills = [s for s in skills_flat if s['classification'] == "Partial"]
        missing_skills = [s for s in skills_flat if s['classification'] == "Missing"]

        st.markdown("---")
        c_score, c_stats = st.columns([1, 1.4], gap="medium")
        with c_score:
            st.markdown("<div class='saas-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-weight:700; color:#64748B; margin:0;'>التوافق العام</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:3.2rem; font-weight:800; color:#2563EB; margin:0;'>{overall_score}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_stats:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("#### 📋 عدادات كفاءة المهارات")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.markdown(f"""<div class="stat-card stat-card-green"><div class="stat-number">{len(strong_skills)}</div><div class="stat-label">قوية</div></div>""", unsafe_allow_html=True)
            with s_col2:
                st.markdown(f"""<div class="stat-card stat-card-amber"><div class="stat-number">{len(partial_skills)}</div><div class="stat-label">جزئية</div></div>""", unsafe_allow_html=True)
            with s_col3:
                st.markdown(f"""<div class="stat-card stat-card-red"><div class="stat-number">{len(missing_skills)}</div><div class="stat-label">مفقودة</div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 9. الصفحة 2: التجربة الأكاديمية
# ----------------------------------------------------------------------------
else:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">🧪 التجربة الأكاديمية والمقارنة بين النماذج</div>
            <div class="hero-subtitle">مقارنة TF-IDF مع Sentence Transformers</div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('analyzed_p1', False):
        cv_text = st.session_state['cv_text']
        matched_results = st.session_state['matched_results']
        flat_all = []
        for cat_list in matched_results.values():
            flat_all.extend(cat_list)
            
        if flat_all:
            req_names = [item["name"] for item in flat_all]
            start_time_a = time.time()
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus = [cv_text] + req_names
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_sims = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0:1]).flatten() * 100
            exec_time_a = (time.time() - start_time_a) * 1000

            start_time_b = time.time()
            cv_emb = embed_model.encode(cv_text, convert_to_tensor=True)
            req_embs = embed_model.encode(req_names, convert_to_tensor=True)
            transformer_sims = util.cos_sim(req_embs, cv_emb).flatten().cpu().numpy() * 100
            exec_time_b = (time.time() - start_time_b) * 1000

            st.metric("متوسط TF-IDF", f"{round(float(tfidf_sims.mean()), 1)}%", f"{round(exec_time_a, 1)} ms")
            st.metric("متوسط Sentence Transformers", f"{round(float(transformer_sims.mean()), 1)}%", f"{round(exec_time_b, 1)} ms")
        else:
            st.info("لا توجد بيانات للمقارنة.")
    else:
        st.info("👈 قم بإجراء تحليل السيرة الذاتية في الصفحة الأولى أولاً.")
