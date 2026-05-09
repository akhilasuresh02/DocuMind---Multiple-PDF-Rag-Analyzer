# =============================================================================
# pdfRagAnalyser v3 — Complete UI Redesign + Speed Optimizations
# =============================================================================
# Speed optimizations applied:
#   - Parallel PDF page extraction with ThreadPoolExecutor
#   - Parallel per-PDF chunking
#   - FAISS index cached in st.session_state (no disk reload per query)
#   - BM25Retriever cached in st.session_state (no rebuild per query)
#   - Embedding model preloaded at startup via st.cache_resource
#   - Font preconnect hints added to CSS for faster Google Fonts load
#   - Fixed model name sequence (removed non-existent gemini-3.1-flash-lite-preview)
#   - rewrite_query uses faster model with tighter timeout
#   - FAISS batch size increased for fewer merge passes
#   - CSS @import replaced with <link> preconnect for faster font load
# =============================================================================

import streamlit as st
try:
    from pypdf import PdfReader  # pypdf: maintained successor to PyPDF2, better malformed-PDF handling
except ImportError:
    from PyPDF2 import PdfReader  # fallback if pypdf not installed yet
import pandas as pd
import base64
import os
import re
import time
import html as html_lib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# === LANGCHAIN IMPORTS ===
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

try:
    from langchain_experimental.text_splitter import SemanticChunker as _SemanticChunker  # noqa: F401
except ImportError:
    pass
SEMANTIC_CHUNKER_AVAILABLE = False  # forced off


# =============================================================================
# === CONSTANTS ===
# =============================================================================

BROAD_KEYWORDS = [
    "summary", "summarize", "summarise", "overview", "combined",
    "all papers", "all documents", "compare", "comparison", "differences",
    "similarities", "explain", "what is", "tell me about", "describe",
    "highlight", "key points", "main points", "findings", "conclusion"
]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = "faiss_index"


# =============================================================================
# === HELPER: Detect broad vs specific queries ===
# =============================================================================

def is_broad_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in BROAD_KEYWORDS)


# =============================================================================
# === PDF EXTRACTION — parallel across PDFs, safe sequential page reads ===
# =============================================================================

def _extract_single_pdf(pdf) -> tuple:
    """Extract text from a single PDF.

    Runs in a thread pool (one thread per PDF) but reads pages sequentially
    within each PDF. PdfReader is NOT thread-safe for concurrent page access
    on the same object — parallelising pages triggers PdfReadError on PDFs
    with cross-references or font-width tables (e.g. the b'>' error).

    Per-page errors are caught and skipped so a single bad page does not
    abort the whole document.
    """
    try:
        pdf_reader = PdfReader(pdf, strict=False)
    except TypeError:
        # older PyPDF2 builds don't accept strict kwarg
        pdf_reader = PdfReader(pdf)

    pages_text = []
    for page in pdf_reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # Skip unreadable pages (corrupt font maps, invalid objects, etc.)
            pages_text.append("")

    return "".join(pages_text), pdf.name


def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDFs in parallel (one thread per PDF)."""
    all_texts, all_sources = [], []
    with ThreadPoolExecutor(max_workers=min(4, len(pdf_docs))) as ex:
        futures = {ex.submit(_extract_single_pdf, pdf): pdf for pdf in pdf_docs}
        for future in as_completed(futures):
            try:
                text, name = future.result()
            except Exception as e:
                pdf = futures[future]
                text, name = "", getattr(pdf, "name", "unknown.pdf")
                st.warning(f"Could not read '{name}': {e}. Skipping.")
            all_texts.append(text)
            all_sources.append(name)
    return all_texts, all_sources


# =============================================================================
# === CHUNKING — parallelized per PDF ===
# =============================================================================

def _chunk_single(text_source_pair, embeddings):
    """Chunk a single (text, source) pair. Runs in thread pool."""
    text, source = text_source_pair
    if not text.strip():
        return [], []

    chunks = None

    if SEMANTIC_CHUNKER_AVAILABLE:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
            chunks = splitter.split_text(text)
            if len(chunks) < 10:
                chunks = None
        except Exception:
            chunks = None

    if chunks is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

    metadata = [{"source": source, "chunk_id": i} for i in range(len(chunks))]
    return chunks, metadata


def get_text_chunks(all_texts, all_sources):
    """Chunk all PDFs in parallel."""
    embeddings = get_local_embeddings()
    all_chunks, all_metadata = [], []

    with ThreadPoolExecutor(max_workers=min(4, len(all_texts))) as ex:
        futures = [ex.submit(_chunk_single, (t, s), embeddings)
                   for t, s in zip(all_texts, all_sources)]
        for future in as_completed(futures):
            chunks, metadata = future.result()
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    return all_chunks, all_metadata


# =============================================================================
# === EMBEDDINGS — cached for server lifetime ===
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    """Load embedding model once and cache it for the lifetime of the server."""
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


# =============================================================================
# === VECTOR STORE — larger batches, faster merges ===
# =============================================================================

def get_vector_store(text_chunks, metadata, batch_size=64):
    """Build FAISS index in larger batches — fewer merge passes = faster."""
    embeddings = get_local_embeddings()
    try:
        vector_store = None
        total = len(text_chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_vs = FAISS.from_texts(
                text_chunks[start:end],
                embedding=embeddings,
                metadatas=metadata[start:end]
            )
            if vector_store is None:
                vector_store = batch_vs
            else:
                vector_store.merge_from(batch_vs)

        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store, text_chunks
    except Exception as e:
        raise RuntimeError(f"Error building vector store: {e}") from e


# =============================================================================
# === HYBRID RETRIEVAL — cached BM25 + cached FAISS ===
# =============================================================================

def get_bm25_retriever(text_chunks: list) -> BM25Retriever:
    """Build (or return cached) BM25 retriever. Rebuilt only when chunks change."""
    cache_key = "_bm25_retriever"
    cached_key = "_bm25_chunk_hash"
    chunk_hash = hash(tuple(text_chunks[:10]))  # fast hash of first 10 chunks

    if (cache_key not in st.session_state or
            st.session_state.get(cached_key) != chunk_hash):
        retriever = BM25Retriever.from_texts(text_chunks)
        st.session_state[cache_key] = retriever
        st.session_state[cached_key] = chunk_hash

    return st.session_state[cache_key]


def get_cached_faiss() -> FAISS:
    """Return FAISS index from session cache — avoids disk reload per query."""
    if "_faiss_vs" not in st.session_state:
        embeddings = get_local_embeddings()
        st.session_state["_faiss_vs"] = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings,
            allow_dangerous_deserialization=True
        )
    return st.session_state["_faiss_vs"]


def get_hybrid_docs(text_chunks, vector_store, query, k=10):
    """Manual hybrid: BM25 keyword + FAISS semantic, deduplicated."""
    bm25_retriever = get_bm25_retriever(text_chunks)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query)

    vs = get_cached_faiss()
    faiss_docs = vs.similarity_search(query, k=k)

    seen = set()
    merged = []
    for doc in faiss_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    for doc in bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged[:k]


# =============================================================================
# === QUERY REWRITING — session-cached, fast model ===
# =============================================================================

def rewrite_query(user_question: str, api_key: str) -> str:
    """Rewrite user question using a lightweight model with exponential backoff."""
    cache_key = f"_rw_{user_question.strip().lower()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    rewrite_prompt = (
        f"Rewrite this question into a better document search query. "
        f"Be specific, expand abbreviations, keep it under 20 words. "
        f"Return ONLY the rewritten query, nothing else.\n\n"
        f"Question: {user_question}\nRewritten query:"
    )

    # Use gemini-1.5-flash for rewriting — fastest available, low quota cost
    for attempt in range(3):
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                max_retries=1,
                request_timeout=8,
            )
            result = model.predict(rewrite_prompt).strip()
            st.session_state[cache_key] = result
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(2 ** (attempt + 1))
            else:
                break

    st.session_state[cache_key] = user_question
    return user_question


# =============================================================================
# === PROMPT TEMPLATES ===
# =============================================================================

STUFF_PROMPT_TEMPLATE = """
You are an expert analyst specializing in financial reports, research papers, and technical documents.

Instructions:
- Answer based on the provided context
- For summaries: synthesize ALL context into structured sections with headers and bullet points
- For follow-ups: use chat_history to resolve references like "it", "they", "this", "that"
- Think step by step before answering complex questions
- If context is partially insufficient, answer what you CAN and clearly state what is missing
- NEVER say "answer not available" if there is ANY relevant information — always provide partial insight
- Format answers with clear headers and bullet points for readability
- Specifically analyze in financial documents:
    * Financial statements and key ratios
    * Related party transactions
    * KMP (Key Management Personnel) remuneration changes
    * Any signs of financial irregularities or red flags

Chat History: {chat_history}
Context: {context}
Question: {question}

Think step by step:
Answer:
"""

MAP_REDUCE_PROMPT_TEMPLATE = """
You are an expert document analyst. Summarize and analyze the following document section thoroughly.
Extract all key information including financial data, findings, and important details.

Context: {context}
Question: {question}

Detailed analysis:
"""


# =============================================================================
# === CONVERSATIONAL CHAIN ===
# =============================================================================

def get_conversational_chain(api_key: str, chain_type: str = "stuff", model_name: str = "gemini-2.0-flash"):
    """Build LangChain QA chain with the given model."""
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key,
        max_retries=2,
        request_timeout=60,
    )
    if chain_type == "map_reduce":
        prompt = PromptTemplate(
            template=MAP_REDUCE_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(model, chain_type="map_reduce", question_prompt=prompt)
    else:
        prompt = PromptTemplate(
            template=STUFF_PROMPT_TEMPLATE,
            input_variables=["context", "question", "chat_history"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def run_chain_with_backoff(api_key, chain_type, docs, user_question, history_text, max_attempts=3):
    """Run the QA chain with exponential backoff on 429 / 503 errors.
    Uses valid, fast models in order of speed.
    """
    # Valid model sequence: fastest → most capable fallback
    model_sequence = [
        "gemini-2.0-flash",    # fast, production-ready
        "gemini-2.5-flash",    # more capable fallback
        "gemini-1.5-flash",    # stable high-quota fallback
    ]

    last_error = None
    for model_name in model_sequence:
        for attempt in range(max_attempts):
            try:
                chain = get_conversational_chain(api_key, chain_type=chain_type, model_name=model_name)
                if chain_type == "map_reduce":
                    return chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                else:
                    return chain(
                        {"input_documents": docs, "question": user_question, "chat_history": history_text},
                        return_only_outputs=True
                    )
            except Exception as e:
                err = str(e)
                last_error = e
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 5 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                elif any(code in err for code in ["503", "500", "UNAVAILABLE", "overloaded", "INTERNAL"]):
                    time.sleep(2)
                    break
                else:
                    raise

    raise RuntimeError(
        f"All models and retries exhausted. Last error: {last_error}. "
        "Please wait a moment and try again."
    )


# =============================================================================
# === V3 GLOBAL CSS — Design System (fonts via <link> for faster load) ===
# =============================================================================

# Font preconnect + preload injected separately for faster initial render
FONT_PRELOAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

GLOBAL_CSS = """
<style>
/* ──────────────────────────────────────────────────────────────────────────
   DESIGN TOKENS
   ────────────────────────────────────────────────────────────────────────── */
:root {
  --t0: #B0E0E6;
  --t1: #5FB3D5;
  --t2: #17A2B8;
  --t3: #0D7A8A;
  --t4: #0A5A6B;
  --tx: #0F172A;
  --ts: #334155;
  --tt: #64748B;
  --bg: #F8FAFC;
  --bg2: #EFF6FA;
  --ok: #10B981;
  --warn: #F59E0B;
  --err: #EF4444;
  --hl: #06B6D4;
  --chat-bg: #0A3D47;
  --shadow-card: 0 2px 12px rgba(13,122,138,.08);
  --shadow-hover: 0 8px 24px rgba(13,122,138,.16);
  --glow-sm: 0 0 12px rgba(23,162,184,.35);
  --glow-md: 0 0 22px rgba(6,182,212,.45);
}

/* ──────────────────────────────────────────────────────────────────────────
   KEYFRAME ANIMATIONS
   ────────────────────────────────────────────────────────────────────────── */
@keyframes breathe {
  0%,100% { opacity:.55; transform:scale(1); }
  50%      { opacity:1;   transform:scale(1.06); }
}
@keyframes pulse {
  0%,100% { opacity:1; }
  50%     { opacity:.4; }
}
@keyframes slideInDown {
  from { opacity:0; transform:translateY(-22px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes slideInLeft {
  from { opacity:0; transform:translateX(-28px); }
  to   { opacity:1; transform:translateX(0); }
}
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes scaleIn {
  from { opacity:0; transform:scale(.91); }
  to   { opacity:1; transform:scale(1); }
}
@keyframes gradientFlow {
  0%   { background-position:0%   50%; }
  50%  { background-position:100% 50%; }
  100% { background-position:0%   50%; }
}
@keyframes shimmer {
  0%   { transform:translateX(-120%); }
  100% { transform:translateX(220%); }
}
@keyframes glow {
  0%,100% { box-shadow:0 0 8px  rgba(6,182,212,.25); }
  50%      { box-shadow:0 0 22px rgba(6,182,212,.60); }
}
@keyframes borderPulse {
  0%,100% { border-color:var(--t2); box-shadow:0 0 6px  rgba(23,162,184,.30); }
  50%      { border-color:var(--hl); box-shadow:0 0 18px rgba(6,182,212,.55); }
}
@keyframes dotBounce {
  0%,80%,100% { transform:scale(.55); opacity:.35; }
  40%          { transform:scale(1);   opacity:1;   }
}
@keyframes spin    { from { transform:rotate(0deg);   } to { transform:rotate(360deg);  } }
@keyframes spinRev { from { transform:rotate(360deg); } to { transform:rotate(0deg);    } }

@media (prefers-reduced-motion:reduce) {
  *, *::before, *::after {
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    transition-duration:.01ms !important;
  }
}

/* ──────────────────────────────────────────────────────────────────────────
   GLOBAL
   ────────────────────────────────────────────────────────────────────────── */
* { box-sizing:border-box; }
.stApp {
  background: var(--bg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   HEADER
   ────────────────────────────────────────────────────────────────────────── */
header[data-testid="stHeader"] {
  background: linear-gradient(90deg, var(--t4) 0%, var(--t3) 100%) !important;
  box-shadow: 0 4px 20px rgba(13,122,138,.28);
  animation: slideInDown .4s ease-out;
}

/* ──────────────────────────────────────────────────────────────────────────
   HIDE STREAMLIT CHROME
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"]         { display:none !important; }
#MainMenu                              { display:none !important; }
[data-testid="stMainMenu"]             { display:none !important; }
[data-testid="stToolbar"] [kind="secondary"],
[data-testid="stAppDeployButton"]      { display:none !important; }
footer                                 { display:none !important; }

/* ──────────────────────────────────────────────────────────────────────────
   CUSTOM SCROLLBAR
   ────────────────────────────────────────────────────────────────────────── */
::-webkit-scrollbar           { width:6px; }
::-webkit-scrollbar-track     { background:transparent; }
::-webkit-scrollbar-thumb     { background:rgba(23,162,184,.28); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:rgba(23,162,184,.58); }

/* ──────────────────────────────────────────────────────────────────────────
   SIDEBAR
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg) !important;
  border-right: 2px solid var(--t2) !important;
  animation: slideInLeft .3s ease-out;
}
[data-testid="stSidebar"] > div:first-child { padding-top:.9rem !important; }
[data-testid="stSidebar"] .stMarkdown p { color:var(--tx) !important; }

/* ──────────────────────────────────────────────────────────────────────────
   SIDEBAR CUSTOM COMPONENTS
   ────────────────────────────────────────────────────────────────────────── */
.v3-logo {
  display:flex; align-items:center; gap:.55rem;
  padding-bottom:.7rem; margin-bottom:.4rem;
  border-bottom:1px solid rgba(176,224,230,.35);
}
.v3-logo-icon  { font-size:1.45rem; }
.v3-logo-text  { font-family:'IBM Plex Sans',sans-serif; font-size:.95rem; font-weight:700; color:var(--t4); }
.v3-logo-badge {
  font-family:'IBM Plex Mono',monospace; font-size:.58rem; font-weight:600;
  background:var(--t2); color:white;
  padding:2px 7px; border-radius:4px; margin-left:.2rem; vertical-align:middle;
}

.v3-api-row {
  display:flex; align-items:center; gap:.5rem;
  padding:.42rem .72rem;
  background:var(--bg2); border-radius:8px;
  border:1px solid rgba(23,162,184,.2); margin-bottom:.72rem;
}
.v3-api-dot {
  width:8px; height:8px; border-radius:50%; flex-shrink:0;
  background:var(--ok);
  animation:pulse 2.5s cubic-bezier(.4,0,.6,1) infinite;
}
.v3-api-dot.err { background:var(--err); }
.v3-api-label { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--ts); }

.v3-section-label {
  font-family:'IBM Plex Sans',sans-serif; font-size:.63rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.5px; color:var(--ts);
  margin:.85rem 0 .38rem;
}

.v3-status {
  display:flex; align-items:center; gap:.5rem;
  padding:.42rem .72rem; border-radius:8px;
  font-family:'IBM Plex Sans',sans-serif; font-size:.8rem; font-weight:600;
  margin:.45rem 0;
}
.v3-status.ready { background:rgba(16,185,129,.1); color:#047857; border:1px solid rgba(16,185,129,.28); }
.v3-status.idle  { background:var(--bg2); color:var(--tt); border:1px solid rgba(176,224,230,.38); }

/* ──────────────────────────────────────────────────────────────────────────
   BUTTONS
   ────────────────────────────────────────────────────────────────────────── */
.stButton > button {
  font-family:'IBM Plex Sans',sans-serif !important;
  font-weight:600 !important; font-size:.84rem !important;
  border-radius:8px !important; border:none !important;
  background:var(--t2) !important; color:white !important;
  box-shadow:0 4px 12px rgba(23,162,184,.22) !important;
  transition:all .2s ease !important;
  animation:scaleIn .2s ease-out;
}
.stButton > button:hover  { background:var(--t3) !important; box-shadow:0 8px 20px rgba(23,162,184,.32) !important; transform:translateY(-2px) !important; }
.stButton > button:active { background:var(--t4) !important; transform:translateY(0) !important; box-shadow:0 2px 8px rgba(23,162,184,.18) !important; }
.stButton > button:disabled { opacity:.5 !important; cursor:not-allowed !important; transform:none !important; }

/* ──────────────────────────────────────────────────────────────────────────
   FILE UPLOADER
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background:var(--bg2) !important;
  border:2px dashed var(--t1) !important;
  border-radius:12px !important;
  transition:all .3s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  background:#E0F2FE !important;
  border-color:var(--t2) !important;
  box-shadow:0 0 20px rgba(23,162,184,.18) !important;
}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
  font-family:'IBM Plex Sans',sans-serif !important;
  color:#0F172A !important;
  -webkit-text-fill-color:#0F172A !important;
}
[data-testid="stFileUploaderDropzone"] small {
  color:#334155 !important;
  -webkit-text-fill-color:#334155 !important;
}
[data-testid="stFileUploaderFile"] {
  background:white !important;
  border:1px solid rgba(23,162,184,.3) !important;
  border-radius:8px !important;
  animation:fadeInUp .3s ease-out;
  transition:background .2s ease !important;
}
[data-testid="stFileUploaderFile"]:hover { background:var(--bg2) !important; }
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] div {
  color:#0F172A !important;
  -webkit-text-fill-color:#0F172A !important;
}
[data-testid="stFileUploaderFile"] small {
  color:#334155 !important;
  -webkit-text-fill-color:#334155 !important;
}
[data-testid="stFileUploaderFile"] button,
[data-testid="stFileUploaderFile"] button svg,
[data-testid="stFileUploaderFile"] [kind="secondary"] {
  color:var(--t2) !important;
  fill:var(--t2) !important;
  -webkit-text-fill-color:var(--t2) !important;
  border-color:var(--t1) !important;
  background:transparent !important;
}
[data-testid="stFileUploaderFile"] button:hover {
  color:var(--t3) !important;
  fill:var(--t3) !important;
  background:rgba(23,162,184,.1) !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   DOWNLOAD BUTTON
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
  background:linear-gradient(135deg, var(--t4), var(--t3)) !important;
  font-size:.8rem !important;
}
[data-testid="stDownloadButton"] > button:hover { filter:brightness(1.1) !important; transform:translateY(-1px) !important; }

/* ──────────────────────────────────────────────────────────────────────────
   CHAT INPUT
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stChatInput"],
.stChatInput {
  border:2px solid var(--t2) !important;
  border-radius:16px !important;
  box-shadow:0 4px 16px rgba(13,122,138,.18) !important;
  transition:all .2s ease !important;
  background:var(--chat-bg) !important;
}
[data-testid="stChatInput"]:focus-within,
.stChatInput:focus-within {
  border-color:var(--hl) !important;
  box-shadow:var(--glow-md) !important;
  animation:borderPulse 1.3s ease-in-out infinite;
}
[data-testid="stChatInputTextArea"],
.stChatInput textarea,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div[contenteditable],
[data-testid="stChatInput"] p,
[data-testid="stChatInput"] * {
  font-family:'IBM Plex Sans',sans-serif !important;
  font-size:.94rem !important;
  color:#FFFFFF !important;
  -webkit-text-fill-color:#FFFFFF !important;
  caret-color:#5FB3D5 !important;
  background:transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] p[data-placeholder]::before,
[data-testid="stChatInput"] [data-placeholder]::before {
  color:rgba(176,224,230,.7) !important;
  -webkit-text-fill-color:rgba(176,224,230,.7) !important;
  opacity:1 !important;
}
[data-testid="stBottom"],
.stBottomBlockContainer {
  background:linear-gradient(to top, rgba(10,58,71,.85) 0%, rgba(10,58,71,.0) 100%) !important;
  padding-bottom:.8rem !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   SPINNER
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div { border-top-color:var(--t2) !important; }
[data-testid="stSpinner"] p {
  font-family:'IBM Plex Sans',sans-serif !important;
  color:var(--t3) !important; font-size:.88rem !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   ALERTS / STATUS
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius:10px !important;
  font-family:'IBM Plex Sans',sans-serif !important;
  font-size:.85rem !important;
  animation:fadeInUp .3s ease-out;
}
[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
  -webkit-text-fill-color:unset !important;
}
.stSuccess, [data-baseweb="notification"][kind="positive"] {
  background:rgba(16,185,129,.12) !important;
  border-left:4px solid #10B981 !important;
}
.stSuccess *, .stSuccess p { color:#064E3B !important; -webkit-text-fill-color:#064E3B !important; }
.stWarning, [data-baseweb="notification"][kind="warning"] {
  background:rgba(245,158,11,.1) !important;
  border-left:4px solid #F59E0B !important;
}
.stWarning *, .stWarning p { color:#78350F !important; -webkit-text-fill-color:#78350F !important; }
.stError, [data-baseweb="notification"][kind="negative"] {
  background:rgba(239,68,68,.1) !important;
  border-left:4px solid #EF4444 !important;
}
.stError *, .stError p { color:#7F1D1D !important; -webkit-text-fill-color:#7F1D1D !important; }
.stInfo, [data-baseweb="notification"][kind="info"] {
  background:rgba(23,162,184,.1) !important;
  border-left:4px solid var(--t2) !important;
}
.stInfo *, .stInfo p { color:var(--t4) !important; -webkit-text-fill-color:var(--t4) !important; }

/* ──────────────────────────────────────────────────────────────────────────
   DIVIDER & CAPTION
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stDivider"] hr { border-color:rgba(176,224,230,.35) !important; }
.stCaption p {
  font-family:'IBM Plex Mono',monospace !important;
  font-size:.72rem !important;
  color:#334155 !important;
  -webkit-text-fill-color:#334155 !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   HERO TITLE
   ────────────────────────────────────────────────────────────────────────── */
.v3-hero { padding:.7rem 0 .9rem; animation:fadeInUp .5s ease-out; }
.v3-hero-title {
  font-family:'IBM Plex Sans',sans-serif;
  font-size:1.85rem; font-weight:700; margin:0 0 .18rem;
  background:linear-gradient(135deg, var(--t2) 0%, var(--t0) 40%, var(--hl) 75%, var(--t2) 100%);
  background-size:280% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation:gradientFlow 5s ease infinite;
}
.v3-hero-sub {
  font-family:'IBM Plex Mono',monospace;
  font-size:.71rem; color:var(--tt); letter-spacing:.25px;
}

/* ──────────────────────────────────────────────────────────────────────────
   CHAT WRAP
   ────────────────────────────────────────────────────────────────────────── */
.v3-chat-wrap {
  font-family:'IBM Plex Sans',sans-serif;
  max-width:860px; margin:0 auto; padding-bottom:1.5rem;
}
.v3-bubble {
  display:flex; gap:.78rem; align-items:flex-start;
  margin-bottom:1.05rem;
  animation:fadeInUp .32s ease-out;
}
.v3-av {
  flex-shrink:0; width:36px; height:36px; border-radius:9px; overflow:hidden;
  box-shadow:0 2px 8px rgba(13,122,138,.18);
}
.v3-av img { width:100%; height:100%; object-fit:cover; }
.v3-av-bot {
  flex-shrink:0; width:36px; height:36px; border-radius:9px;
  background:linear-gradient(135deg, var(--t2), var(--t3));
  display:flex; align-items:center; justify-content:center; font-size:1.1rem;
  box-shadow:0 2px 8px rgba(13,122,138,.22);
}
.v3-body { flex:1; min-width:0; }
.v3-user .v3-content {
  background:linear-gradient(135deg, var(--t2) 0%, var(--t3) 100%);
  color:white;
  padding:.82rem 1.08rem;
  border-radius:4px 16px 16px 16px;
  font-size:.93rem; line-height:1.65;
  box-shadow:0 4px 16px rgba(13,122,138,.24);
  transition:transform .25s ease, box-shadow .25s ease;
  position:relative; overflow:hidden;
}
.v3-user .v3-content::after {
  content:'';
  position:absolute; top:0; left:-90%; width:45%; height:100%;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.09), transparent);
  animation:shimmer 4.5s ease-in-out infinite;
  pointer-events:none;
}
.v3-user .v3-content:hover {
  transform:rotate(.4deg) scale(1.008);
  box-shadow:0 8px 24px rgba(13,122,138,.32);
}
.v3-bot .v3-content {
  background:#F0F9FC;
  color:var(--tx);
  padding:.82rem 1.08rem;
  border-radius:16px 4px 16px 16px;
  border-left:4px solid var(--t2);
  font-size:.93rem; line-height:1.72;
  box-shadow:var(--shadow-card);
  transition:background .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.v3-bot .v3-content:hover {
  background:#E0F2FE;
  box-shadow:0 6px 20px rgba(23,162,184,.14);
  border-left-color:var(--hl);
}
.v3-bot .v3-content strong { color:var(--t3); }
.v3-bot .v3-content code {
  font-family:'IBM Plex Mono',monospace; font-size:.82em;
  background:rgba(23,162,184,.1); padding:1px 5px; border-radius:3px; color:var(--t3);
}
.v3-meta {
  font-family:'IBM Plex Mono',monospace; font-size:.67rem;
  color:var(--tt); margin-top:.28rem;
  opacity:0; transition:opacity .2s ease;
}
.v3-bubble:hover .v3-meta { opacity:1; }
.v3-search-note {
  font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  color:rgba(255,255,255,.82);
  background:rgba(255,255,255,.13);
  padding:3px 10px; border-radius:6px;
  display:inline-block; margin-bottom:.48rem;
  border:1px solid rgba(255,255,255,.2);
}
.v3-source {
  display:inline-flex; align-items:center; gap:.3rem;
  background:rgba(16,185,129,.1); color:#047857;
  font-family:'IBM Plex Mono',monospace; font-size:.7rem; font-weight:500;
  padding:3px 10px; border-radius:16px;
  border:1px solid rgba(16,185,129,.25);
  margin-top:.68rem;
}

/* ──────────────────────────────────────────────────────────────────────────
   EMPTY STATE / LANDING
   ────────────────────────────────────────────────────────────────────────── */
.v3-landing {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:60vh; padding:4rem 2rem; text-align:center;
  animation:fadeInUp .55s ease-out;
}
.v3-landing-headline {
  font-family:'IBM Plex Sans',sans-serif;
  font-size:2.6rem; font-weight:700; line-height:1.2;
  margin:0 0 1rem;
  background:linear-gradient(135deg, var(--t3) 0%, var(--t2) 50%, var(--hl) 100%);
  background-size:200% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation:gradientFlow 4s ease infinite;
}
.v3-landing-sub {
  font-family:'IBM Plex Sans',sans-serif;
  font-size:1.05rem; color:#334155;
  -webkit-text-fill-color:#334155;
  max-width:480px; line-height:1.7; margin:0;
}
.v3-landing-icon {
  font-size:3.8rem; margin-bottom:1.2rem;
  animation:breathe 3.5s ease-in-out infinite;
  filter:drop-shadow(0 4px 12px rgba(23,162,184,.3));
}
.v3-pills { display:flex; flex-wrap:wrap; gap:.4rem; justify-content:center; margin-top:.9rem; }
.v3-pill  {
  font-family:'IBM Plex Mono',monospace; font-size:.67rem;
  background:var(--bg2); color:var(--t3);
  padding:3px 10px; border-radius:12px;
  border:1px solid rgba(23,162,184,.22);
}
</style>
"""


# =============================================================================
# === MARKDOWN HELPER ===
# =============================================================================

def md_to_html(text: str) -> str:
    """Convert basic markdown in bot answers to styled HTML."""
    out = html_lib.escape(text)
    out = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', out)
    out = re.sub(r'\*(?!\*)(.+?)\*(?!\*)', r'<em>\1</em>', out)
    out = re.sub(r'`(.+?)`', r'<code>\1</code>', out)
    out = re.sub(r'^#### (.+)$',
                 r'<strong style="color:var(--t2)">\1</strong>',
                 out, flags=re.MULTILINE)
    out = re.sub(r'^### (.+)$',
                 r'<strong style="display:block;color:var(--t3);margin:.5em 0 .18em">\1</strong>',
                 out, flags=re.MULTILINE)
    out = re.sub(r'^## (.+)$',
                 r'<strong style="display:block;color:var(--t4);font-size:1.03em;margin:.6em 0 .2em">\1</strong>',
                 out, flags=re.MULTILINE)
    out = re.sub(r'^# (.+)$',
                 r'<strong style="display:block;color:var(--t4);font-size:1.07em;margin:.7em 0 .25em">\1</strong>',
                 out, flags=re.MULTILINE)
    out = re.sub(r'^[\*\-] (.+)$',
                 r'<span style="display:block;padding-left:.85em">• \1</span>',
                 out, flags=re.MULTILINE)
    out = re.sub(r'^(\d+)\. (.+)$',
                 r'<span style="display:block;padding-left:.85em">\1. \2</span>',
                 out, flags=re.MULTILINE)
    out = out.replace('\n', '<br>')
    return out


# =============================================================================
# === UI RENDERING ===
# =============================================================================

def render_message_pair(question, answer, rewritten=None, sources=None, is_latest=False):
    safe_q  = html_lib.escape(question)
    bot_html = md_to_html(answer)

    search_note = ""
    if rewritten and rewritten.lower().strip() != question.lower().strip() and is_latest:
        safe_r = html_lib.escape(rewritten)
        search_note = f'<span class="v3-search-note">🔍 Searched for: {safe_r}</span><br>'

    source_html = ""
    if sources and is_latest:
        safe_s = html_lib.escape(sources)
        source_html = f'<div class="v3-source">📄 {safe_s}</div>'

    return f"""
<div class="v3-bubble v3-user">
  <div class="v3-av"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png" alt="User"></div>
  <div class="v3-body">
    <div class="v3-content">{search_note}{safe_q}</div>
  </div>
</div>
<div class="v3-bubble v3-bot">
  <div class="v3-av-bot" aria-label="AI assistant">🤖</div>
  <div class="v3-body">
    <div class="v3-content">{bot_html}{source_html}</div>
  </div>
</div>
"""


def render_chat_ui(conversation_history, rewritten_query="", original_question=""):
    st.markdown('<div class="v3-chat-wrap">', unsafe_allow_html=True)

    if conversation_history:
        latest = conversation_history[-1]
        q, a = latest[0], latest[1]
        sources = latest[5] if len(latest) > 5 else None
        st.markdown(
            render_message_pair(q, a, rewritten=rewritten_query, sources=sources, is_latest=True),
            unsafe_allow_html=True
        )
        for item in reversed(conversation_history[:-1]):
            q, a = item[0], item[1]
            st.markdown(render_message_pair(q, a), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# === MAIN USER INPUT HANDLER ===
# =============================================================================

def user_input(user_question, api_key, pdf_docs, conversation_history, vector_store, text_chunks):
    if vector_store is None or not text_chunks:
        st.warning("Please upload PDF files and click 'Submit & Process' before asking questions.")
        return

    broad = is_broad_query(user_question)
    k = 20 if broad else 8
    chain_type = "stuff"

    # Step 1: Rewrite query
    with st.spinner("🔍 Optimising search query..."):
        rewritten_query = rewrite_query(user_question, api_key)

    # Step 2: Hybrid retrieval (uses cached FAISS + cached BM25)
    with st.spinner("📚 Retrieving relevant chunks..."):
        try:
            docs = get_hybrid_docs(text_chunks, None, rewritten_query, k=k)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return

    # Step 3: Build chat history string (last 5 turns only)
    history_text = ""
    for q, a, *_ in conversation_history[-5:]:
        history_text += f"Human: {q}\nAssistant: {a}\n\n"

    # Step 4: Run chain with automatic backoff + model fallback
    with st.spinner("Thinking..."):
        try:
            response = run_chain_with_backoff(
                api_key, chain_type, docs, user_question, history_text
            )
        except RuntimeError as e:
            st.error(f"❌ {e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return

    response_text = response.get("output_text", "No response generated.")

    # Step 5: Extract sources
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in docs if hasattr(doc, "metadata")
    })
    sources_display = ", ".join(sources) if sources else "Unknown"

    # Step 6: Store in history
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append((
        user_question,
        response_text,
        "Google AI",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ", ".join(pdf_names),
        sources_display
    ))

    # Step 7: Update session state for persistent render
    st.session_state.last_rewritten = rewritten_query
    st.session_state.last_question  = user_question

    # Step 8: Render chat
    render_chat_ui(conversation_history, rewritten_query, user_question)


# =============================================================================
# === MAIN APP ===
# =============================================================================

def main():
    st.set_page_config(
        page_title="DocuMind",
        page_icon="🧠",
        layout="wide"
    )

    # ── Font preconnect (faster than @import inside <style>) ─────────────
    st.markdown(FONT_PRELOAD_HTML, unsafe_allow_html=True)

    # ── Inject global design system CSS ──────────────────────────────────
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Warm up embedding model in background on first load ──────────────
    # This triggers the @st.cache_resource once so the first query is fast
    get_local_embeddings()

    # ── Hero title ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="v3-hero">
      <div class="v3-hero-title">🧠 DocuMind</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    # ── Session state init ────────────────────────────────────────────────
    defaults = {
        'conversation_history': [],
        'vector_store_ready': False,
        'text_chunks': [],
        'last_rewritten': "",
        'last_question': "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:

        st.markdown("""
        <div class="v3-logo">
          <span class="v3-logo-icon">🧠</span>
          <span class="v3-logo-text">DocuMind<span class="v3-logo-badge">AI</span></span>
        </div>
        """, unsafe_allow_html=True)

        if not api_key:
            st.markdown("""
            <div class="v3-api-row">
              <div class="v3-api-dot err"></div>
              <span class="v3-api-label">No API Key Detected</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="v3-section-label">📁 Documents</div>', unsafe_allow_html=True)

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
            help="Upload one or more PDF files (max 200 MB each). Click Submit & Process after uploading.",
            label_visibility="collapsed"
        )

        if st.button("⚙️  Submit & Process", use_container_width=True, type="primary"):
            if pdf_docs:
                with st.spinner("📖 Extracting text from PDFs..."):
                    all_texts, all_sources = get_pdf_text(pdf_docs)
                with st.spinner("✂️ Chunking text..."):
                    chunks, metadata = get_text_chunks(all_texts, all_sources)
                with st.spinner("🧠 Building FAISS vector index..."):
                    try:
                        vs, stored_chunks = get_vector_store(chunks, metadata)
                        # Cache FAISS in session to avoid disk reload per query
                        st.session_state["_faiss_vs"] = vs
                        st.session_state.vector_store_ready = True
                        st.session_state.pdf_docs = pdf_docs
                        st.session_state.text_chunks = stored_chunks
                        # Invalidate BM25 cache so it rebuilds with new chunks
                        st.session_state.pop("_bm25_retriever", None)
                        st.session_state.pop("_bm25_chunk_hash", None)
                        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pdf_docs)} PDF(s).")
                        chunker_used = "SemanticChunker" if SEMANTIC_CHUNKER_AVAILABLE else "RecursiveCharacterTextSplitter"
                        st.caption(f"Chunker: `{chunker_used}` · Retrieval: `Hybrid BM25 + FAISS`")
                    except RuntimeError as e:
                        st.error(str(e))
            else:
                st.warning("Upload at least one PDF file first.")

        if st.session_state.vector_store_ready:
            n = len(st.session_state.text_chunks)
            st.markdown(f'<div class="v3-status ready">🟢 {n} chunks indexed &amp; ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="v3-status idle">⚪ No documents processed yet</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="v3-section-label">⚙️ Controls</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        if col1.button("↺ Undo Last", use_container_width=True):
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()
            st.rerun()
        if col2.button("🔄 Reset All", use_container_width=True):
            for key in ['conversation_history', 'vector_store_ready', 'text_chunks',
                        'last_rewritten', 'last_question', '_faiss_vs',
                        '_bm25_retriever', '_bm25_chunk_hash']:
                st.session_state.pop(key, None)
            st.session_state['conversation_history'] = []
            st.session_state['vector_store_ready'] = False
            st.session_state['text_chunks'] = []
            st.session_state['last_rewritten'] = ""
            st.session_state['last_question'] = ""
            st.rerun()

        if st.session_state.conversation_history:
            st.divider()
            st.markdown('<div class="v3-section-label">⬇ Downloads</div>', unsafe_allow_html=True)
            df = pd.DataFrame(
                st.session_state.conversation_history,
                columns=["Question", "Answer", "Model", "Timestamp", "PDF Name", "Sources"]
            )
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download Chat History (CSV)",
                data=csv_bytes,
                file_name="documind_chat.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ── MAIN CHAT AREA ────────────────────────────────────────────────────
    if not api_key:
        st.error(
            "⚠️ Google API Key not found! "
            "Add `GOOGLE_API_KEY=your_key` to your `.env` file and restart the app."
        )
        st.stop()

    user_question = st.chat_input(
        "💬 Ask a question about your documents...",
        key="chat_input"
    )

    if not user_question:
        if st.session_state.conversation_history:
            render_chat_ui(
                st.session_state.conversation_history,
                st.session_state.last_rewritten,
                st.session_state.last_question
            )
        else:
            st.markdown("""
            <div class="v3-landing">
              <div class="v3-landing-icon">🧠</div>
              <div class="v3-landing-headline">Stop reading.<br>Start asking.</div>
              <p class="v3-landing-sub">Upload any PDF and ask it questions like you&rsquo;re talking to an expert.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        if not st.session_state.vector_store_ready:
            st.warning("⚠️ Please upload your PDFs and click **Submit & Process** first.")
        else:
            pdf_docs    = st.session_state.get('pdf_docs')
            text_chunks = st.session_state.get('text_chunks', [])
            user_input(
                user_question,
                api_key,
                pdf_docs,
                st.session_state.conversation_history,
                st.session_state.vector_store_ready,
                text_chunks
            )


if __name__ == "__main__":
    main()