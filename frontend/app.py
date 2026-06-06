"""
Phase 8 — Streamlit Frontend
Run from project root: streamlit run frontend/app.py
Requires the FastAPI server running on http://127.0.0.1:8000
"""

import sys
from pathlib import Path

import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import frontend.api_client as api

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engineering RAG",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Engineering RAG")
page = st.sidebar.radio("Navigate", ["💬 Chat", "📄 Documents", "📊 Status"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def server_ok() -> bool:
    try:
        api.health()
        return True
    except Exception:
        return False


_QUERY_TYPE_LABELS = {
    "parts":       ("🔩", "Parts Manual"),
    "maintenance": ("🔧", "Maintenance Manual"),
    "both":        ("🔩🔧", "Parts + Maintenance Manuals"),
}


def _query_type_badge(query_type: str | None, collections: list | None = None):
    """Render a small caption badge showing which collection(s) were searched."""
    if not query_type:
        return
    icon, label = _QUERY_TYPE_LABELS.get(query_type, ("", query_type))
    detail = f" (`{'`, `'.join(collections)}`)" if collections else ""
    st.caption(f"{icon} Searched: **{label}**{detail}")


def _banner(ok: bool):
    if ok:
        st.sidebar.success("Server: online")
    else:
        st.sidebar.error("Server: offline — start with `uvicorn app.main:app --reload --port 8000`")


_banner(server_ok())

# ── Login form ────────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar.expander("Login (required for uploads)", expanded=st.session_state.token is None):
    if st.session_state.token:
        st.success("Logged in")
        if st.button("Log out"):
            st.session_state.token = None
            st.rerun()
    else:
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                st.session_state.token = api.login(username, password)
                st.success("Login successful!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.title("💬 Chat with your Engineering Manuals")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.sidebar.expander("Settings", expanded=False):
        k = st.slider("Chunks to retrieve (k)", 1, 20, 6)
        model = st.selectbox("LLM model", ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"])
        advanced_mode = st.toggle(
            "Advanced mode",
            value=False,
            help="Enables query rewriting + cross-encoder reranking (slower, higher precision)",
        )
        show_sources = st.checkbox("Show source chunks", value=True)
        show_contexts = st.checkbox("Show retrieved context", value=False)

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _query_type_badge(msg.get("query_type"), msg.get("collections_searched"))
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        parts = [
                            s.get("source_file", ""),
                            s.get("section_title", ""),
                            f"p.{s['page_start']}" if s.get("page_start") else "",
                        ]
                        st.markdown("- " + " — ".join(p for p in parts if p))
            if msg["role"] == "assistant" and show_contexts and msg.get("contexts"):
                with st.expander("Retrieved context"):
                    for i, ctx in enumerate(msg["contexts"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(ctx[:600] + ("…" if len(ctx) > 600 else ""))
            # ── Feedback buttons ──────────────────────────────────────────
            if msg["role"] == "assistant" and msg.get("msg_id") is not None:
                mid = msg["msg_id"]
                if msg.get("feedback_given"):
                    st.caption("✅ Feedback recorded")
                else:
                    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
                    with fb_col1:
                        if st.button("👍", key=f"up_{mid}"):
                            try:
                                api.submit_feedback(msg["question"], msg["content"], "positive", sources=msg.get("sources", []))
                                msg["feedback_given"] = True
                                st.rerun()
                            except Exception:
                                st.warning("Could not save feedback.")
                    with fb_col2:
                        if st.button("👎", key=f"dn_{mid}"):
                            try:
                                api.submit_feedback(msg["question"], msg["content"], "negative", sources=msg.get("sources", []))
                                msg["feedback_given"] = True
                                st.rerun()
                            except Exception:
                                st.warning("Could not save feedback.")

    # ── Input ─────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your engineering manuals…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            spinner_msg = "Rewriting query & reranking…" if advanced_mode else "Thinking…"
            with st.spinner(spinner_msg):
                try:
                    result = (
                        api.ask_advanced(prompt, k=k, model=model)
                        if advanced_mode
                        else api.ask(prompt, k=k, model=model)
                    )
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources", [])
                    query_type = result.get("query_type")
                    collections_searched = result.get("collections_searched")

                    # Retrieve context chunks separately for display
                    raw_chunks = api.query(prompt, k=k)
                    contexts = [c.get("text", "") for c in raw_chunks]

                    st.markdown(answer)
                    _query_type_badge(query_type, collections_searched)

                    if show_sources and sources:
                        with st.expander("Sources"):
                            for s in sources:
                                parts = [
                                    s.get("source_file", ""),
                                    s.get("section_title", ""),
                                    f"p.{s['page_start']}" if s.get("page_start") else "",
                                ]
                                st.markdown("- " + " — ".join(p for p in parts if p))

                    if show_contexts and contexts:
                        with st.expander("Retrieved context"):
                            for i, ctx in enumerate(contexts, 1):
                                st.markdown(f"**Chunk {i}**")
                                st.text(ctx[:600] + ("…" if len(ctx) > 600 else ""))

                    msg_id = len(st.session_state.messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "contexts": contexts,
                        "query_type": query_type,
                        "collections_searched": collections_searched,
                        "msg_id": msg_id,
                        "question": prompt,
                        "feedback_given": False,
                    })

                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

    if st.sidebar.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Documents":
    st.title("📄 Document Management")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Upload manuals")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        total = len(uploaded_files)
        st.info(
            f"**{total} file{'s' if total > 1 else ''} selected:** "
            + ", ".join(f.name for f in uploaded_files)
        )

        if st.button("Upload & Process All", type="primary"):
            if not st.session_state.token:
                st.error("Please log in first (sidebar → Login).")
            else:
                token = st.session_state.token
                succeeded, failed = 0, 0

                for idx, uploaded in enumerate(uploaded_files, start=1):
                    st.markdown(f"**[{idx}/{total}] {uploaded.name}**")
                    progress = st.progress(0, text="Uploading…")
                    try:
                        # Step 1: Upload + parse
                        upload_result = api.upload_manual(uploaded.read(), uploaded.name, token)
                        progress.progress(33, text="Chunking…")

                        # Step 2: Chunk
                        chunk_result = api.chunk_manual(upload_result["saved_json"], token)
                        progress.progress(66, text="Indexing…")

                        # Step 3: Index
                        index_result = api.index_manual(chunk_result["chunked_file"], token)
                        progress.progress(100, text="Done!")

                        manual_type = upload_result.get("manual_type", "maintenance")
                        icon, type_label = _QUERY_TYPE_LABELS.get(manual_type, ("📄", manual_type))
                        st.success(
                            f"✅ {chunk_result['chunk_count']} chunks indexed into "
                            f"{index_result.get('collection_name', 'unknown')}  {icon} {type_label}"
                        )
                        succeeded += 1

                    except Exception as e:
                        progress.empty()
                        st.error(f"Failed: {e}")
                        failed += 1

                st.divider()
                if failed == 0:
                    st.success(f"All {succeeded} file{'s' if succeeded > 1 else ''} processed successfully.")
                else:
                    st.warning(f"{succeeded} succeeded, {failed} failed.")

    st.divider()

    # ── List ──────────────────────────────────────────────────────────────────
    st.subheader("Indexed manuals")

    if st.button("Refresh list"):
        st.rerun()

    try:
        manuals = api.list_manuals()
        if not manuals:
            st.info("No manuals indexed yet. Upload a PDF above.")
        else:
            for m in manuals:
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{m.get('filename', 'Unknown')}**")
                        st.caption(
                            f"ID: `{m.get('manual_id', '')}` | "
                            f"Elements: {m.get('element_count', '?')}"
                        )
                    with c2:
                        btn1, btn2 = st.columns(2)
                        with btn1:
                            if st.button("⚙ Index", key=f"idx_{m['manual_id']}", help="Chunk & index into vector DB"):
                                if not st.session_state.token:
                                    st.error("Please log in first.")
                                else:
                                    try:
                                        token = st.session_state.token
                                        manual_id = m["manual_id"]
                                        parsed_path = f"storage/parsed/{manual_id}.json"
                                        with st.spinner("Chunking…"):
                                            chunk_result = api.chunk_manual(parsed_path, token)
                                        with st.spinner("Indexing…"):
                                            api.index_manual(chunk_result["chunked_file"], token)
                                        st.success(f"Indexed {chunk_result['chunk_count']} chunks.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Index failed: {e}")
                        with btn2:
                            if st.button("🗑", key=f"del_{m['manual_id']}", help="Delete manual"):
                                if not st.session_state.token:
                                    st.error("Please log in first.")
                                else:
                                    try:
                                        api.delete_manual(m["manual_id"], st.session_state.token)
                                        st.success("Deleted.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {e}")

    except Exception as e:
        st.error(f"Could not fetch manuals: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Status":
    st.title("📊 System Status")

    col1, col2 = st.columns(2)

    # ── Server health ─────────────────────────────────────────────────────────
    with col1:
        st.subheader("API Server")
        try:
            health = api.health()
            st.success(f"Online — `{health}`")
        except Exception as e:
            st.error(f"Offline: {e}")
            st.code("uvicorn app.main:app --reload --port 8000", language="bash")

    # ── Document stats ────────────────────────────────────────────────────────
    with col2:
        st.subheader("Document Store")
        try:
            manuals = api.list_manuals()
            total_elements = sum(m.get("element_count", 0) for m in manuals)
            st.metric("Manuals indexed", len(manuals))
            st.metric("Total parsed elements", total_elements)
        except Exception as e:
            st.error(f"Could not load stats: {e}")

    st.divider()

    # ── Manual breakdown ──────────────────────────────────────────────────────
    st.subheader("Manual breakdown")
    try:
        manuals = api.list_manuals()
        if manuals:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Filename": m.get("filename", ""),
                    "Elements": m.get("element_count", 0),
                    "Manual ID": m.get("manual_id", ""),
                }
                for m in manuals
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No manuals indexed yet.")
    except Exception as e:
        st.error(f"Error: {e}")

    st.divider()

    # ── Feedback summary ──────────────────────────────────────────────────────
    st.subheader("User Feedback")
    try:
        fb = api.get_feedback()
        total = fb.get("count", 0)
        if total == 0:
            st.info("No feedback yet. Ask questions and use 👍/👎 in the Chat page.")
        else:
            pos = fb.get("positive", 0)
            neg = fb.get("negative", 0)
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Total Ratings", total)
            fc2.metric("👍 Positive", pos)
            fc3.metric("👎 Negative", neg)

            if neg > 0:
                with st.expander(f"Negative feedback ({neg})"):
                    for entry in fb["feedback"]:
                        if entry["rating"] == "negative":
                            st.markdown(f"**Q:** {entry['question']}")
                            st.caption(f"{entry['timestamp']} | Comment: {entry.get('comment') or '—'}")
                            st.divider()
    except Exception as e:
        st.error(f"Could not load feedback: {e}")

    st.divider()

    # ── RAGAS results ─────────────────────────────────────────────────────────
    st.subheader("Latest RAGAS Evaluation")
    results_dir = Path("evaluation/results")
    result_files = sorted(results_dir.glob("ragas_*.json"), reverse=True) if results_dir.exists() else []

    if not result_files:
        st.info("No evaluation results found. Run `python evaluation/run_ragas.py`.")
    else:
        import json
        latest = result_files[0]
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        meta_col, score_col = st.columns([1, 2])

        with meta_col:
            st.markdown(f"**Run:** `{data.get('timestamp', '')}`")
            st.markdown(f"**Model:** `{data.get('model', '')}`")
            st.markdown(f"**Samples:** {data.get('sample_count', '')}")
            st.markdown(f"**k:** {data.get('k', '')}")

        with score_col:
            overall = data.get("overall_average", 0)
            colour = "normal" if overall >= 0.7 else "inverse"
            st.metric("Overall Average", f"{overall:.4f}", delta=None)
            for metric, val in scores.items():
                st.metric(metric.replace("_", " ").title(), f"{val:.4f}")

        if len(result_files) > 1:
            with st.expander(f"Previous runs ({len(result_files) - 1})"):
                for f_path in result_files[1:]:
                    with open(f_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    st.markdown(
                        f"`{d.get('timestamp')}` — Overall: **{d.get('overall_average', '?')}** "
                        f"| Samples: {d.get('sample_count')} | k={d.get('k')}"
                    )
