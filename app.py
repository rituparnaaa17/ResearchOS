
import os
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchOS",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# GLOBAL CSS  – ResearchOS design system
# ============================================================

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Top header bar ── */
    .ros-header {
        padding: 18px 0 14px 0;
        border-bottom: 1px solid #e2e2e2;
        margin-bottom: 24px;
    }
    .ros-title {
        font-size: 52px;
        font-weight: 800;
        color: #9ca3af;
        margin: 0;
        letter-spacing: 4px;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .ros-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin: 2px 0 0 0;
    }

    /* ── Card ── */
    .ros-card {
        border: 1px solid #e2e2e2;
        border-radius: 6px;
        padding: 18px 20px;
        background: #ffffff;
        margin-bottom: 16px;
        height: 100%;
    }
    .ros-card-title {
        font-size: 14px;
        font-weight: 600;
        color: #111111;
        margin: 0 0 4px 0;
    }
    .ros-card-label {
        font-size: 11px;
        color: #9ca3af;
        margin: 0 0 10px 0;
    }
    .ros-card-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
    }
    .ros-card-header-right {
        font-size: 11px;
        color: #9ca3af;
    }

    /* ── Textarea (simulated) ── */
    .stTextArea textarea {
        border: 1px solid #e2e2e2 !important;
        border-radius: 4px !important;
        background: #f9f9f9 !important;
        font-size: 13px !important;
        color: #374151 !important;
        font-family: 'Inter', sans-serif !important;
        resize: vertical !important;
    }
    .stTextArea textarea:focus {
        border-color: #6b7280 !important;
        box-shadow: none !important;
    }

    /* ── Text input ── */
    .stTextInput input {
        border: 1px solid #e2e2e2 !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 5px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    /* Primary button (Generate Summary) */
    div[data-testid="column"]:first-child .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }
    div[data-testid="column"]:first-child .stButton > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
    /* Secondary button (Analyze Similarity) */
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background-color: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button:hover {
        background-color: #f3f4f6 !important;
    }

    /* ── Similarity table ── */
    .ros-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .ros-table th {
        background: #f3f4f6;
        color: #374151;
        font-weight: 600;
        text-align: left;
        padding: 7px 10px;
        border: 1px solid #e2e2e2;
    }
    .ros-table td {
        padding: 7px 10px;
        border: 1px solid #e2e2e2;
        color: #374151;
        vertical-align: top;
    }
    .ros-table tr:nth-child(even) td { background: #fafafa; }
    .ros-table-empty {
        font-size: 12px;
        color: #9ca3af;
        text-align: center;
        padding: 20px 0;
    }
    .ros-table-note {
        font-size: 10.5px;
        color: #9ca3af;
        margin-top: 6px;
    }

    /* ── Summary panel ── */
    .ros-summary-box {
        background: #f9fafb;
        border: 1px solid #e2e2e2;
        border-radius: 4px;
        padding: 12px 14px;
        font-size: 13px;
        color: #374151;
        line-height: 1.6;
        min-height: 90px;
        white-space: pre-wrap;
    }
    .ros-summary-empty {
        color: #9ca3af;
        font-style: italic;
    }

    /* ── Results section ── */
    .ros-section-title {
        font-size: 15px;
        font-weight: 600;
        color: #111111;
        margin: 28px 0 14px 0;
        padding-top: 20px;
        border-top: 1px solid #e2e2e2;
    }
    .ros-paper-card {
        border: 1px solid #e2e2e2;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 12px;
        background: #ffffff;
    }
    .ros-paper-title {
        font-size: 14px;
        font-weight: 600;
        color: #111111;
        margin: 0 0 6px 0;
    }
    .ros-paper-meta {
        font-size: 12px;
        color: #6b7280;
        margin: 2px 0;
    }
    .ros-paper-score {
        display: inline-block;
        font-size: 11.5px;
        font-weight: 600;
        color: #2563eb;
        background: #eff6ff;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 6px 0 8px 0;
    }
    .ros-paper-abstract {
        font-size: 12.5px;
        color: #374151;
        line-height: 1.6;
        margin: 8px 0 10px 0;
    }
    .ros-paper-link a {
        font-size: 12.5px;
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }
    .ros-paper-link a:hover { text-decoration: underline; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #d1d5db !important;
        border-radius: 5px !important;
        padding: 6px !important;
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] {
        font-size: 12px !important;
    }

    /* ── Divider ── */
    hr { border-color: #e2e2e2 !important; }

    /* ── Info / warning ── */
    .stAlert { font-size: 13px !important; }

    /* ── Reduce block gap ── */
    .block-container { padding-top: 0 !important; padding-bottom: 32px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="ros-header">
        <p class="ros-title">ResearchOS</p>
        <p class="ros-subtitle">Research Paper Summarization and Similarity Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATASET  (unchanged logic)
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_HERE, "arXiv_scientific_dataset - Copy.csv")

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.error(
        f"Dataset not found. Make sure this file is in the same folder as app.py:\n{CSV_FILE}"
    )
    st.stop()
except Exception as exc:
    st.error(f"Error loading dataset: {exc}")
    st.stop()


# ============================================================
# CLEAN DATASET  (unchanged logic)
# ============================================================

df.columns = df.columns.str.strip()

for col in ["title", "summary"]:
    if col not in df.columns:
        df[col] = ""

for col in df.columns:
    df[col] = df[col].fillna("")

df["title"] = df["title"].astype(str)
df["summary"] = df["summary"].astype(str)
df["search_text"] = df["title"] + " " + df["summary"]


# ============================================================
# TF-IDF MODEL  (unchanged logic, unchanged cache decorator)
# ============================================================

@st.cache_resource
def create_tfidf_model(text_data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(text_data)
    return vectorizer, matrix


try:
    vectorizer, tfidf_matrix = create_tfidf_model(
        tuple(df["search_text"].tolist())
    )
except Exception as exc:
    st.error(f"Could not create TF-IDF model: {exc}")
    st.stop()


# ============================================================
# SESSION STATE  – store results between button clicks
# ============================================================

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "query_used" not in st.session_state:
    st.session_state["query_used"] = ""
if "abstract_text" not in st.session_state:
    st.session_state["abstract_text"] = ""


# ============================================================
# HELPER – build arXiv link  (unchanged logic)
# ============================================================

def _build_paper_link(row):
    possible_link_cols = [
        "link", "url", "paper_url", "pdf_url",
        "arxiv_url", "Link", "URL", "Paper Link", "paper_link",
    ]
    for col in possible_link_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                val = str(val).strip()
                if val.startswith("http://") or val.startswith("https://"):
                    return val

    paper_id = row.get("id", "")
    if pd.notna(paper_id):
        paper_id = str(paper_id).strip()
    else:
        paper_id = ""

    if not paper_id:
        return None

    if paper_id.startswith("abs-"):
        paper_id = paper_id[4:]
    paper_id = paper_id.strip()

    if paper_id.startswith("http://") or paper_id.startswith("https://"):
        return paper_id

    paper_id = re.sub(r"^/?abs/", "", paper_id)
    return f"https://arxiv.org/abs/{paper_id}"


# ============================================================
# HELPER – run similarity search  (unchanged ML logic)
# ============================================================

def _run_search(query_text, n_papers):
    query_vector = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-n_papers:][::-1]
    res = df.iloc[top_idx].copy()
    res["similarity_score"] = scores[top_idx]
    res = res.reset_index(drop=True)
    return res


# ============================================================
# HELPER – extract PDF text  (unchanged logic)
# ============================================================

def _extract_pdf_text(uploaded):
    if PdfReader is None:
        st.error("PyPDF2 is not installed. Run: pip install PyPDF2")
        return None
    try:
        reader = PdfReader(uploaded)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
        return text[:10000]
    except Exception as exc:
        st.error(f"Could not read the PDF: {exc}")
        return None


# ============================================================
# TWO-COLUMN DASHBOARD
# ============================================================

left_col, right_col = st.columns([1, 1], gap="medium")

# -----------------------------------------------------------
# LEFT COLUMN – Research Paper Abstract
# -----------------------------------------------------------

with left_col:
    st.markdown(
        """
        <div class="ros-card">
            <div class="ros-card-header">
                <p class="ros-card-title">Research Paper Abstract</p>
            </div>
            <p class="ros-card-label">Enter a research topic or upload a PDF to analyse</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Abstract / query textarea
    abstract_input = st.text_area(
        "Research topic or abstract",
        value=st.session_state["abstract_text"],
        height=210,
        placeholder=(
            "Type a research topic or paste an abstract here…\n\n"
            "Example: A low-cost sensor network for monitoring air quality in urban areas."
        ),
        label_visibility="collapsed",
    )

    # PDF upload
    uploaded_file = st.file_uploader(
        "Or upload a PDF",
        type=["pdf"],
        help="Upload a research paper PDF; its text will be used as the query.",
        label_visibility="visible",
    )

    # Number of results slider
    number_of_papers = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Buttons row
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        gen_summary_btn = st.button(
            "Generate Summary",
            use_container_width=True,
            key="btn_gen_summary",
        )

    with btn_col2:
        analyze_btn = st.button(
            "Analyze Similarity",
            use_container_width=True,
            key="btn_analyze",
        )

    st.markdown(
        '<p style="font-size:11px;color:#9ca3af;margin-top:6px;">'
        "Interface illustration; no measured results are shown until you search."
        "</p>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
# RIGHT COLUMN – Similarity Analysis + Generated Summary
# -----------------------------------------------------------

with right_col:

    # ── Top card: Textual Similarity Analysis ──────────────

    results_df = st.session_state.get("results_df")
    query_used = st.session_state.get("query_used", "")

    # Build table HTML
    if results_df is not None and len(results_df) > 0:
        rows_html = ""
        for _, row in results_df.iterrows():
            title_a = str(row.get("title", "Untitled"))[:60]
            score = float(row.get("similarity_score", 0))
            rows_html += (
                f"<tr>"
                f"<td>{query_used[:40]}{'…' if len(query_used) > 40 else ''}</td>"
                f"<td>{title_a}{'…' if len(str(row.get('title',''))) > 60 else ''}</td>"
                f"<td>{score:.4f}</td>"
                f"</tr>"
            )
        table_note = (
            "Query-to-paper cosine similarity (TF-IDF). "
            "Scores reflect textual overlap; they do not establish plagiarism."
        )
        table_body = f"""
            <table class="ros-table">
                <thead>
                    <tr>
                        <th>Paper A (Query)</th>
                        <th>Paper B (Result)</th>
                        <th>Cosine Similarity</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            <p class="ros-table-note">{table_note}</p>
        """
    else:
        table_body = (
            '<p class="ros-table-empty">'
            "No results yet. Enter a topic and click <em>Analyze Similarity</em>."
            "</p>"
        )

    st.markdown(
        f"""
        <div class="ros-card">
            <div class="ros-card-header">
                <p class="ros-card-title">Textual Similarity Analysis</p>
                <span class="ros-card-header-right">
                    {"Live results" if results_df is not None else "Awaiting query"}
                </span>
            </div>
            {table_body}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Bottom card: Generated Summary ─────────────────────

    if results_df is not None and len(results_df) > 0:
        top_row = results_df.iloc[0]
        top_title = str(top_row.get("title", "Untitled"))
        top_summary = str(top_row.get("summary", "Abstract not available."))
        summary_label = "Abstract of top-ranked paper (no generative summarisation in this pipeline)"
        summary_content = f"<strong>{top_title}</strong>\n\n{top_summary}"
        is_empty = False
    else:
        summary_content = ""
        summary_label = "Illustrative output"
        is_empty = True

    if is_empty:
        summary_html = '<span class="ros-summary-empty">Run a similarity search to see the top paper\'s abstract here.</span>'
    else:
        summary_html = summary_content.replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="ros-card">
            <div class="ros-card-header">
                <p class="ros-card-title">Generated Summary</p>
                <span class="ros-card-header-right">{summary_label}</span>
            </div>
            <div class="ros-summary-box">{summary_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BUTTON ACTIONS  (after layout so columns are already rendered)
# ============================================================

def _resolve_query():
    """Return the active query string from textarea or PDF, or None."""
    if uploaded_file is not None:
        text = _extract_pdf_text(uploaded_file)
        if text:
            return text.strip()
        return None
    q = abstract_input.strip()
    return q if q else None


# ── Analyze Similarity ──────────────────────────────────────
if analyze_btn:
    query_text = _resolve_query()
    if not query_text:
        st.warning("Please enter a research topic or upload a PDF before analysing.")
    else:
        with st.spinner("Running TF-IDF similarity search…"):
            try:
                res = _run_search(query_text, number_of_papers)
                st.session_state["results_df"] = res
                st.session_state["query_used"] = query_text[:80]
                st.session_state["abstract_text"] = abstract_input
                st.rerun()
            except Exception as exc:
                st.error(f"Error while searching: {exc}")


# ── Generate Summary ─────────────────────────────────────────
if gen_summary_btn:
    query_text = _resolve_query()
    if not query_text:
        st.warning("Please enter a research topic or upload a PDF first.")
    else:
        with st.spinner("Finding the most relevant paper…"):
            try:
                # The pipeline does not include a generative model;
                # we retrieve the top-1 match and surface its abstract.
                res = _run_search(query_text, number_of_papers)
                st.session_state["results_df"] = res
                st.session_state["query_used"] = query_text[:80]
                st.session_state["abstract_text"] = abstract_input
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")


# ============================================================
# RESULTS SECTION
# ============================================================

results_df = st.session_state.get("results_df")

if results_df is not None and len(results_df) > 0:

    st.markdown(
        '<p class="ros-section-title">Relevant Research Papers</p>',
        unsafe_allow_html=True,
    )

    for rank, (_, row) in enumerate(results_df.iterrows(), start=1):

        title = str(row.get("title", "Untitled Paper"))
        category = str(row.get("category", ""))
        authors = str(row.get("authors", ""))
        published_date = str(row.get("published_date", ""))
        score = float(row.get("similarity_score", 0))
        summary = str(row.get("summary", "Abstract not available."))
        paper_link = _build_paper_link(row)

        # Meta line
        meta_parts = []
        if category:
            meta_parts.append(f"<strong>Category:</strong> {category}")
        if authors:
            meta_parts.append(f"<strong>Authors:</strong> {authors}")
        if published_date:
            meta_parts.append(f"<strong>Published:</strong> {published_date}")
        meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

        # Link HTML
        if paper_link:
            link_html = (
                f'<div class="ros-paper-link">'
                f'<a href="{paper_link}" target="_blank">Read Paper →</a>'
                f"</div>"
            )
        else:
            link_html = (
                '<p style="font-size:12px;color:#9ca3af;">Link not available.</p>'
            )

        st.markdown(
            f"""
            <div class="ros-paper-card">
                <p class="ros-paper-title">{rank}. {title}</p>
                <p class="ros-paper-meta">{meta_html}</p>
                <span class="ros-paper-score">Similarity: {score:.4f}</span>
                <p class="ros-paper-abstract">{summary}</p>
                {link_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

elif results_df is not None and len(results_df) == 0:
    st.info("No research papers matched your query.")