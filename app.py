# app.py
"""
Streamlit demo app:
"سامانه پیشنهاد مقالات مشابه / پیشنهاد داور (دموی پروپوزال)"

نکات:
- این نسخه برای دموی پروپوزال طراحی شده: داده‌ها از Semantic Scholar گرفته می‌شوند.
- در نسخه نهایی توصیه می‌شود داده‌ها را از Scopus/PubMed و مدل‌های domain-specific fine-tune شده استفاده کنید.
"""

import time
import requests
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import plotly.express as px

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_AUTHOR = "https://api.semanticscholar.org/graph/v1/author/{}"
SEARCH_LIMIT_DEFAULT = 50  # برای دموی اولیه
REQUEST_SLEEP = 0.2  # احترام به rate-limit (تنظیم قابل تغییر)

# ---------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------
st.set_page_config(page_title="دموی پیشنهاد داور/مقالات مشابه", layout="wide")
st.title("🧭 سامانه پیشنهاد مقالات مشابه — دموی پروپوزال")
st.markdown(
    """
این نسخه‌ی نمونه از **Sentence Transformers** برای پیدا کردن مقالات مشابه استفاده می‌کند.
داده‌ها از Semantic Scholar گرفته می‌شوند (برای دموی اولیه). در نسخه‌ی نهایی از **Scopus API** استفاده خواهد شد.
"""
)

# ---------------------------------------------------------------------
# Helpers: مدل‌ها و API calls
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bi_encoder(prefer_specter: bool = True):
    """بارگذاری مدل bi-encoder (سعی می‌کند SPECTER را استفاده کند؛ در صورت عدم‌دسترسی از MPNet استفاده می‌کند)."""
    tried = []
    if prefer_specter:
        try:
            st.info("بارگذاری مدل SPECTER (خصوصیته مقالات علمی)...")
            model = SentenceTransformer("allenai-specter")
            return model
        except Exception as e:
            tried.append(("allenai-specter", str(e)))

    # fallback
    try:
        st.info("بارگذاری مدل all-mpnet-base-v2 (فول بک).")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        return model
    except Exception as e:
        tried.append(("all-mpnet-base-v2", str(e)))
        # last fallback to MiniLM
    try:
        st.info("بارگذاری مدل all-MiniLM-L6-v2 (کم‌حجم).")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception as e:
        tried.append(("all-MiniLM-L6-v2", str(e)))
        raise RuntimeError(f"بارگذاری مدل شکست خورد. تلاش‌های انجام‌شده: {tried}")

@st.cache_resource(show_spinner=False)
def load_reranker():
    """بارگذاری cross-encoder reranker اختیاری (سبک)"""
    try:
        from sentence_transformers import CrossEncoder
        # مدل سریع و سبک MS-MARCO style برای reranking
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return reranker
    except Exception as e:
        st.warning("بارگذاری reranker موفق نبود؛ reranker غیرفعال می‌شود.")
        return None

@st.cache_data(ttl=3600)
def semantic_scholar_search(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """جستجوی اولیه در Semantic Scholar (metadata)"""
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors.name,authors.authorId,publicationDate,venue,url"
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=20)
        time.sleep(REQUEST_SLEEP)
        if resp.status_code == 200:
            return resp.json().get("data", [])
        else:
            st.error(f"Semantic Scholar API returned status {resp.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"خطا در ارتباط با Semantic Scholar: {e}")
        return []

@st.cache_data(ttl=3600)
def get_author_profile(author_id: str) -> Dict[str, Any]:
    """در صورت نیاز اطلاعات بیشتر نویسنده را می‌گیرد (h-index, paperCount)"""
    if not author_id:
        return {}
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_AUTHOR.format(author_id), params={"fields": "name,hIndex,paperCount,affiliations"}, timeout=10)
        time.sleep(REQUEST_SLEEP)
        if resp.status_code == 200:
            return resp.json()
        return {}
    except requests.exceptions.RequestException:
        return {}

# ---------------------------------------------------------------------
# UI: تنظیمات کاربر
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("تنظیمات دموی پروپوزال")
    prefer_specter = st.checkbox("استفاده از مدل SPECTER (اگر موجود باشد)", value=True)
    use_reranker = st.checkbox("استفاده از Reranker برای بهبود نتایج (اختیاری)", value=True)
    top_k = st.slider("تعداد نتایج بالایی که نمایش داده شود (Top K)", min_value=5, max_value=30, value=10)
    search_limit = st.slider("حداکثر مقاله‌های جستجوی اولیه (Semantic Scholar)", min_value=10, max_value=100, value=SEARCH_LIMIT_DEFAULT)
    similarity_threshold = st.slider("آستانه شباهت برای شمارش 'مقالات مرتبط' (cosine)", 0.0, 1.0, 0.30, 0.01)
    st.markdown("---")
    st.caption("نکته: این یک نسخه‌ی نمایشی است. داده‌های نسخه نهایی از Scopus/PubMed گرفته خواهند شد.")

# بارگذاری مدل‌ها
model = load_bi_encoder(prefer_specter=prefer_specter)
reranker = load_reranker() if use_reranker else None

# ---------------------------------------------------------------------
# Main UI: ورودی کاربر
# ---------------------------------------------------------------------
st.subheader("چکیده یا عنوان پروپوزال را وارد کنید")
user_query = st.text_area("چکیده یا عنوان", height=200, placeholder="مثال: 'Machine learning methods for predicting drug solubility'")

col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button("🔎 پیدا کردن مقالات مشابه")
with col2:
    st.write("")
    st.write("")

# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------
if run_button:
    if not user_query or user_query.strip() == "":
        st.warning("لطفاً ابتدا یک چکیده یا عنوان وارد کنید.")
    else:
        with st.spinner("در حال اجرا — دریافت مقالات و محاسبه شباهت‌ها..."):
            # 1) جستجوی اولیه
            initial_papers = semantic_scholar_search(user_query, limit=search_limit)
            if not initial_papers:
                st.error("مقاله‌ای پیدا نشد یا خطا در دریافت داده‌ها.")
            else:
                st.success(f"{len(initial_papers)} مقاله از Semantic Scholar دریافت شد (نمونه).")
                # 2) آماده‌سازی دیکشنری و متون برای embedding
                paper_texts = []
                paper_meta = []
                for p in initial_papers:
                    title = p.get("title") or ""
                    abstract = p.get("abstract") or ""
                    # اگر abstract نیست از title استفاده می‌کنیم
                    text_for_embed = abstract if abstract.strip() else title
                    # به‌عنوان snippet برای نمایش، ترجیحا abstract کوتاه
                    snippet = (abstract[:600] + "...") if abstract else title
                    paper_texts.append(text_for_embed)
                    paper_meta.append({
                        "paperId": p.get("paperId"),
                        "title": title,
                        "snippet": snippet,
                        "authors": p.get("authors", []),
                        "publicationDate": p.get("publicationDate"),
                        "venue": p.get("venue"),
                        "url": p.get("url")
                    })

                # 3) محاسبه embedding ها (batch)
                batch_size = 64
                paper_embeddings = model.encode(paper_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
                query_embedding = model.encode(user_query, convert_to_tensor=True)

                # 4) محاسبه cosine similarities
                cosine_scores = util.cos_sim(query_embedding, paper_embeddings).cpu().numpy().flatten()

                # 5) assemble DataFrame
                df = pd.DataFrame(paper_meta)
                df["similarity"] = cosine_scores
                df = df.sort_values("similarity", ascending=False).reset_index(drop=True)

                # 6) (اختیاری) rerank کردن top_n با cross-encoder برای دقت بیشتر
                if reranker is not None:
                    # بخش top candidates برای rerank
                    rerank_top_n = min(200, len(df))
                    candidates = df.head(rerank_top_n)
                    # ساخت جفت‌ها برای reranker: (query, candidate_text)
                    rerank_inputs = []
                    for i, row in candidates.iterrows():
                        # برای rerank از snippet (یا title+snippet) استفاده می‌کنیم
                        cand_text = (row["title"] + " . " + (row["snippet"] or ""))[:512]
                        rerank_inputs.append((user_query, cand_text))
                    # گرفتن نمرات reranker
                    try:
                        rerank_scores = reranker.predict(rerank_inputs, convert_to_numpy=True, show_progress_bar=False)
                        df.loc[:rerank_top_n-1, "rerank_score"] = rerank_scores
                        # کامپوزیت نهایی: ترکیب cosine و rerank (وزن دلخواه)
                        alpha = 0.5
                        # Normalize rerank scores to 0-1
                        if "rerank_score" in df.columns:
                            arr = df["rerank_score"].fillna(df["rerank_score"].min()).to_numpy()
                            if arr.max() - arr.min() > 1e-6:
                                arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
                            else:
                                arr_norm = np.zeros_like(arr)
                            df["composite"] = alpha * df["similarity"] + (1 - alpha) * arr_norm
                            df = df.sort_values("composite", ascending=False).reset_index(drop=True)
                        else:
                            df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
                    except Exception as e:
                        st.warning(f"خطا در reranker: {e} — نتایج بدون rerank نمایش داده می‌شوند.")
                        df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
                else:
                    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)

                # 7) نمایش نتایج اصلی (top_k)
                st.markdown("## نتایج: مقالات مشابه")
                top_df = df.head(top_k).copy()

                for idx, row in top_df.iterrows():
                    with st.expander(f"#{idx+1} — {row['title']}  (شباهت: {row.get('composite', row['similarity']):.2f})"):
                        st.markdown(f"**عنوان:** [{row['title']}]({row.get('url','')})")
                        st.markdown(f"**نشریه / کنفرانس:** `{row.get('venue')}`  —  **تاریخ:** `{row.get('publicationDate')}`")
                        st.markdown(f"**خلاصه / snippet:**\n\n{row.get('snippet')}")
                        st.markdown("**نویسندگان:**")
                        auths = row.get("authors", [])
                        if auths:
                            for a in auths:
                                name = a.get("name") or "ناشناس"
                                aid = a.get("authorId")
                                if aid:
                                    st.markdown(f"- [{name}](https://www.semanticscholar.org/author/{aid})")
                                else:
                                    st.markdown(f"- {name}")
                        else:
                            st.markdown("- نامشخص")

                # 8) خلاصه آماری و نمودار برای نمایش پروپوزال
                st.markdown("---")
                st.subheader("خلاصه و نمودار شباهت")
                fig = px.bar(x=top_df["title"].apply(lambda t: t[:80]), y=top_df.get("composite", top_df["similarity"]),
                             labels={"x": "عنوان (کوتاه‌شده)", "y": "امتیاز شباهت"}, height=380)
                st.plotly_chart(fig, use_container_width=True)

                # 9) استخراج نویسندگان پیشنهادی: شمارش مقالات مرتبط و میانگین similarity
                st.markdown("---")
                st.subheader("پیشنهاد نویسندگان (از روی مقالات مشابه)")
                # شمارش بر اساس نویسندگان در کل نتایج
                author_stats = {}  # id -> {name, count, avg_sim}
                for i, row in df.iterrows():
                    sim = float(row.get("composite", row["similarity"]))
                    for a in row.get("authors", []):
                        aid = a.get("authorId") or a.get("name")
                        name = a.get("name") or "ناشناس"
                        if not aid:
                            continue
                        if aid not in author_stats:
                            author_stats[aid] = {"name": name, "count": 0, "sum_sim": 0.0}
                        author_stats[aid]["count"] += 1
                        author_stats[aid]["sum_sim"] += sim

                # تبدیل به DataFrame
                authors_list = []
                for aid, stt in author_stats.items():
                    avg_sim = stt["sum_sim"] / stt["count"] if stt["count"] else 0
                    authors_list.append({"authorId": aid, "name": stt["name"], "matched_papers": stt["count"], "avg_similarity": avg_sim})
                authors_df = pd.DataFrame(authors_list)
                if not authors_df.empty:
                    authors_df = authors_df.sort_values(["matched_papers", "avg_similarity"], ascending=[False, False]).reset_index(drop=True)
                    # نمایش table خلاصه
                    st.dataframe(authors_df.head(20))
                    # برای هر نویسنده برتر، لینک به پروفایل و اطلاعات بیشتر را بیاور
                    st.markdown("### جزئیات نویسندگان برتر")
                    for i, arow in authors_df.head(10).iterrows():
                        aid = arow["authorId"]
                        name = arow["name"]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**[{name}](https://www.semanticscholar.org/author/{aid})** — تعداد مقالات مطابق: `{int(arow['matched_papers'])}`, میانگین شباهت: `{arow['avg_similarity']:.2f}`")
                        with col2:
                            if st.button(f"جزئیات {i+1}", key=f"author_detail_{i}"):
                                prof = get_author_profile(aid)
                                if prof:
                                    st.write(prof)
                                else:
                                    st.write("اطلاعات بیشتر در دسترس نیست.")

                else:
                    st.info("نویسنده‌ای از نتایج استخراج نشد.")

                # 10) نکات قابل ارائه در پروپوزال
                st.markdown("---")
                st.subheader("نکاتی که در پروپوزال بنویسید (پیشنهاد)")
                st.markdown(
                    """
- این دموی اولیه با استفاده از **Sentence Transformers** و داده‌های عمومی Semantic Scholar ساخته شده است.
- در نسخه‌ی نهایی داده‌ها از **Scopus API** (با دسترسی دانشگاه) تامین می‌شود که پوشش و دقت حوزه‌ای بالاتری فراهم می‌کند.
- برنامه نهایی شامل: (1) dense retrieval با مدل علمی (SPECTER یا مدل fine-tuned) + (2) cross-encoder reranker + (3) fusion با سیگنال‌های بایگانی (h-index, recency, conflict-of-interest).
- برای افزایش دقت در حوزه‌های تخصصی (مثلاً داروسازی) پیشنهاد می‌شود مدل روی دیتاست domain-specific fine-tune شود.
"""
                )

                st.success("🎯 تمام شد — این نسخه دموی پروپوزال است. برای نسخه‌ی نهایی، آماده‌ام pipeline کامل Scopus-based و fine-tuning را پیاده‌سازی کنم.")
