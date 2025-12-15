import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI # type: ignore
import re
import csv
import json
import datetime
import pdfplumber # type: ignore
import pandas as pd
from ui_theme import apply_theme, hero, themed_panel_start, themed_panel_end

load_dotenv()
# Configure from existing env if present; can be overridden via sidebar input
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ensure_api_key():
    key = st.session_state.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return False
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)
    return True

# Lightweight local embedding to avoid external quotas
class SimpleHashEmbeddings:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str):
        vec = [0.0] * self.dim
        for token in (text or "").lower().split():
            idx = (hash(token) % self.dim)
            vec[idx] += 1.0
        # L2 normalize
        norm = sum(v*v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    # Compatibility for older FAISS adapter expecting a callable
    def __call__(self, text: str):
        return self.embed_query(text)

def get_embeddings_provider():
    provider = st.session_state.get("EMBEDDING_PROVIDER")
    if provider == "Local (No-Quota)":
        return SimpleHashEmbeddings()
    if provider == "OpenAI":
        # Use OpenAI small embedding model for low cost
        if st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
            class OpenAIEmbeddings:
                def embed_documents(self, texts):
                    out = client.embeddings.create(model="text-embedding-3-small", input=texts)
                    return [d.embedding for d in out.data]
                def embed_query(self, text):
                    out = client.embeddings.create(model="text-embedding-3-small", input=text)
                    return out.data[0].embedding
                def __call__(self, text):
                    return self.embed_query(text)
            return OpenAIEmbeddings()
        else:
            st.warning("Set OpenAI API key to use OpenAI embeddings.")
            return SimpleHashEmbeddings()
    # Default to Google if key present, else local
    if ensure_api_key():
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return SimpleHashEmbeddings()

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs or []:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text+= page_text
    return  text



def get_text_chunks(text):
    # Respect sidebar-controlled chunk size for advanced RAG tuning
    chunk_size = st.session_state.get("CHUNK_SIZE", 50000)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def build_vector_store(text_chunks):
    embeddings = get_embeddings_provider()
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error embedding content: {e}")
        if not isinstance(embeddings, SimpleHashEmbeddings):
            st.info("Try switching Embedding Provider to 'Local (No-Quota)' in the sidebar.")
        return None


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    provider = st.session_state.get("LLM_PROVIDER", "Google Gemini")
    if provider == "OpenAI":
        # Low-cost, high-quality OpenAI chat model
        key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=key) if key else OpenAI()
        class OpenAIChatWrapper:
            def invoke(self, prompt_text: str):
                msg = [{"role": "user", "content": prompt_text}]
                resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg)
                return resp.choices[0].message.content
        model = OpenAIChatWrapper()
    else:
        chat_model_name = "models/gemini-1.5-flash-latest"
        model = ChatGoogleGenerativeAI(model=chat_model_name, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return (model, prompt)



def ask_with_index(user_question, index, model_prompt_tuple):
    if not index:
        st.warning("Please upload and process documents first.")
        return ""
    # Retrieval controls
    k = st.session_state.get("RETRIEVAL_K", 4)
    # Boost query with focus terms if provided
    focus_terms = st.session_state.get("FOCUS_TERMS", "")
    boosted_query = user_question
    if focus_terms.strip():
        boosted_query = f"{user_question}\nFocus terms: {focus_terms}"
    docs = index.similarity_search(boosted_query, k=k)
    model, prompt = model_prompt_tuple
    formatted = prompt.format(context="\n".join([d.page_content for d in docs]), question=user_question)
    answer = model.invoke(formatted)
    # Save last evidence for citations
    st.session_state["LAST_EVIDENCE"] = [{"content": d.page_content, **(getattr(d, "metadata", {}) or {})} for d in docs]
    return answer

def fused_evidence_answer(question, left_index, right_index, model_prompt_tuple):
    k = st.session_state.get("RETRIEVAL_K", 4)
    focus_terms = st.session_state.get("FOCUS_TERMS", "")
    boosted_query = question
    if focus_terms.strip():
        boosted_query = f"{question}\nFocus terms: {focus_terms}"
    left_docs = left_index.similarity_search(boosted_query, k=k) if left_index else []
    right_docs = right_index.similarity_search(boosted_query, k=k) if right_index else []
    # Rerank by simple section headings if enabled
    if st.session_state.get("RERANK_BY_HEADINGS"):
        def heading_score(text):
            return 1 if re.search(r"^\s*(Chapter|Section|Clause|Part)\s+\d+", text or "", re.IGNORECASE | re.MULTILINE) else 0
        left_docs = sorted(left_docs, key=lambda d: heading_score(d.page_content), reverse=True)
        right_docs = sorted(right_docs, key=lambda d: heading_score(d.page_content), reverse=True)
    fused = left_docs + right_docs
    model, prompt = model_prompt_tuple
    formatted = prompt.format(context="\n".join([d.page_content for d in fused]), question=question)
    answer = model.invoke(formatted)
    st.session_state["LAST_EVIDENCE"] = [{"content": d.page_content, **(getattr(d, "metadata", {}) or {})} for d in fused]
    return answer

def extract_dates_events(text):
    # Simple date regex for DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY
    date_patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b"
    ]
    events = []
    for line in (text or "").splitlines():
        for pat in date_patterns:
            m = re.search(pat, line)
            if m:
                events.append({"date": m.group(0), "text": line.strip()[:300]})
                break
    return events

def clause_split(text):
    # Split by clause-like headings
    parts = re.split(r"\n\s*(Clause\s+\d+|Section\s+\d+|Article\s+\d+)\s*:\s*", text or "", flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]

def simple_risk_score(clause_text):
    risky_terms = ["warranty", "liability", "termination", "penalty", "indemnify", "arbitration"]
    score = sum(1 for t in risky_terms if t in (clause_text or "").lower())
    return min(10, score)




def main():
    st.set_page_config("Multi-PDF Chat Agent", page_icon = ":scroll:", layout="wide")
    apply_theme()
    hero("Multi-PDF's 📚 - Chat Agent 🤖", "Discover and Chat & compare insights across your PDFs")

    # Initialize dual-panel state
    if "left_index" not in st.session_state:
        st.session_state.left_index = None
    if "right_index" not in st.session_state:
        st.session_state.right_index = None
    if "left_text" not in st.session_state:
        st.session_state.left_text = ""
    if "right_text" not in st.session_state:
        st.session_state.right_text = ""
    model_prompt_tuple = get_conversational_chain()
    col_left, col_right = st.columns(2)
    with col_left:
        themed_panel_start()
        st.subheader("📥 Left Panel: Upload & Chat")
        left_files = st.file_uploader("Drag & drop or browse PDFs (Left)", accept_multiple_files=True, type=["pdf"], key="left_upl")
        if st.button("Process Left", key="process_left"):
            with st.spinner("Indexing left documents..."):
                raw_text = get_pdf_text(left_files)
                st.session_state.left_text = raw_text
                chunks = get_text_chunks(raw_text)
                idx = build_vector_store(chunks)
                st.session_state.left_index = idx
                st.success("Left documents indexed.")
        # Quick templates
        st.caption("Templates: Summarize • Key Points • Action Items • Timeline • Pros/Cons")
        tmpl_cols = st.columns(5)
        templates = ["Summarize", "Key points", "Action items", "Timeline", "Pros and cons"]
        for i, t in enumerate(templates):
            if tmpl_cols[i].button(t, key=f"tmpl_left_{t}"):
                st.session_state["left_q"] = f"{t} for these documents"
        # Chat history UI for Left
        if "CHAT_LEFT" not in st.session_state:
            st.session_state["CHAT_LEFT"] = []
        for msg in st.session_state["CHAT_LEFT"]:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""))
        tone = st.selectbox("Tone", ["professional", "concise", "friendly", "formal"], key="tone_left")
        cite_toggle = st.checkbox("Include citations in answers", value=True, key="cite_left")
        style = st.selectbox("Style", ["bullets", "paragraph", "table"], key="style_left")
        # Suggested prompts deck
        with st.expander("Suggested prompts"):
            sug = [
                "Summarize core themes",
                "Extract obligations and deadlines",
                "List risks and mitigations",
                "Compare pricing and SLAs",
                "Draft email to stakeholders",
            ]
            scols = st.columns(len(sug))
            for i, s in enumerate(sug):
                if scols[i].button(s, key=f"sug_left_{i}"):
                    st.session_state["CHAT_LEFT"].append({"role":"user","content":s})
                    ans = ask_with_index(f"Answer in a {tone} tone, {style} style. {s}", st.session_state.left_index, model_prompt_tuple)
                    st.session_state["CHAT_LEFT"].append({"role":"assistant","content":ans})
                    st.experimental_rerun()
        left_q = st.chat_input("Ask about Left documents…")
        if left_q:
            st.session_state["CHAT_LEFT"].append({"role": "user", "content": left_q})
            ans = ask_with_index(f"Answer in a {tone} tone, {style} style. {left_q}", st.session_state.left_index, model_prompt_tuple)
            if cite_toggle:
                ev = st.session_state.get("LAST_EVIDENCE", [])
                cite_text = "\n\nSources:\n" + "\n".join([f"- Chunk {i+1}" for i, _ in enumerate(ev)]) if ev else ""
            else:
                cite_text = ""
            st.session_state["CHAT_LEFT"].append({"role": "assistant", "content": (ans or "") + cite_text})
            with st.chat_message("assistant"):
                # Streaming simulation chunking for better UX
                full = (ans or "") + cite_text
                chunks = [full[i:i+600] for i in range(0, len(full), 600)] if full else []
                ph = st.empty()
                acc = ""
                for ch in chunks:
                    acc += ch
                    ph.markdown(acc)
            # Quick follow-ups
            sugg_cols = st.columns(3)
            followups = ["Explain further", "Give examples", "List key risks"]
            for i, f in enumerate(followups):
                if sugg_cols[i].button(f, key=f"lfu_{i}"):
                    fq = f"{f.lower()} related to: {left_q}"
                    ans2 = ask_with_index(f"Answer in a {tone} tone. {fq}", st.session_state.left_index, model_prompt_tuple)
                    st.session_state["CHAT_LEFT"].append({"role": "assistant", "content": ans2})
                    st.experimental_rerun()
        regen_cols = st.columns(2)
        if regen_cols[0].button("🔄 Regenerate", key="left_regen") and st.session_state.get("CHAT_LEFT"):
            last_user = next((m for m in reversed(st.session_state["CHAT_LEFT"]) if m["role"]=="user"), None)
            if last_user:
                ans = ask_with_index(f"Answer in a {tone} tone. {last_user['content']}", st.session_state.left_index, model_prompt_tuple)
                st.session_state["CHAT_LEFT"].append({"role": "assistant", "content": ans})
                st.experimental_rerun()
        refine = regen_cols[1].text_input("Refine instruction (Left)")
        if refine and st.button("✨ Apply Refinement", key="apply_refine_left") and st.session_state.get("CHAT_LEFT"):
            last_assist = next((m for m in reversed(st.session_state["CHAT_LEFT"]) if m["role"]=="assistant"), None)
            if last_assist:
                model, prompt = get_conversational_chain()
                refined = model.invoke(f"Refine the answer per: {refine}.\nOriginal Answer:\n{last_assist['content']}")
                st.session_state["CHAT_LEFT"].append({"role": "assistant", "content": refined})
                st.experimental_rerun()
        # Annotations
        note = st.text_area("Add note (Left)", key="note_left")
        if st.button("Save Note (Left)"):
            notes = st.session_state.get("NOTES_LEFT", [])
            notes.append({"q": left_q, "a": (st.session_state["CHAT_LEFT"][-1]["content"] if st.session_state.get("CHAT_LEFT") else ""), "note": note})
            st.session_state["NOTES_LEFT"] = notes
            st.success("Saved note.")
        # Transcript export and pin answer
        tc_left = st.session_state.get("CHAT_LEFT", [])
        if tc_left:
            if st.button("📤 Export Chat Transcript (Left)"):
                md = "\n\n".join([("**User:** " + m["content"]) if m["role"]=="user" else ("**Assistant:** " + m["content"]) for m in tc_left])
                st.download_button("Download Transcript", md.encode("utf-8"), file_name="left_chat.md")
            pin_idx = st.number_input("Pin answer #", min_value=1, max_value=len(tc_left), value=len(tc_left))
            if st.button("📌 Pin Answer", key="pin_left"):
                m = tc_left[pin_idx-1]
                st.session_state.setdefault("PINNED_LEFT", []).append(m)
                st.success("Pinned.")
            themed_panel_end()
    with col_right:
        themed_panel_start()
        st.subheader("📥 Right Panel: Upload & Chat")
        right_files = st.file_uploader("Drag & drop or browse PDFs (Right)", accept_multiple_files=True, type=["pdf"], key="right_upl")
        if st.button("Process Right", key="process_right"):
            with st.spinner("Indexing right documents..."):
                raw_text = get_pdf_text(right_files)
                st.session_state.right_text = raw_text
                chunks = get_text_chunks(raw_text)
                idx = build_vector_store(chunks)
                st.session_state.right_index = idx
                st.success("Right documents indexed.")
        st.caption("Templates: Summarize • Key Points • Action Items • Timeline • Pros/Cons")
        tmpl_cols_r = st.columns(5)
        for i, t in enumerate(templates):
            if tmpl_cols_r[i].button(t, key=f"tmpl_right_{t}"):
                st.session_state["right_q"] = f"{t} for these documents"
        # Chat history UI for Right
        if "CHAT_RIGHT" not in st.session_state:
            st.session_state["CHAT_RIGHT"] = []
        for msg in st.session_state["CHAT_RIGHT"]:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""))
        tone_r = st.selectbox("Tone", ["professional", "concise", "friendly", "formal"], key="tone_right")
        cite_toggle_r = st.checkbox("Include citations in answers", value=True, key="cite_right")
        style_r = st.selectbox("Style", ["bullets", "paragraph", "table"], key="style_right")
        with st.expander("Suggested prompts"):
            sug_r = [
                "Summarize core themes",
                "Extract obligations and deadlines",
                "List risks and mitigations",
                "Compare pricing and SLAs",
                "Draft email to stakeholders",
            ]
            scols_r = st.columns(len(sug_r))
            for i, s in enumerate(sug_r):
                if scols_r[i].button(s, key=f"sug_right_{i}"):
                    st.session_state["CHAT_RIGHT"].append({"role":"user","content":s})
                    ans = ask_with_index(f"Answer in a {tone_r} tone, {style_r} style. {s}", st.session_state.right_index, model_prompt_tuple)
                    st.session_state["CHAT_RIGHT"].append({"role":"assistant","content":ans})
                    st.experimental_rerun()
        right_q = st.chat_input("Ask about Right documents…")
        if right_q:
            st.session_state["CHAT_RIGHT"].append({"role": "user", "content": right_q})
            ans = ask_with_index(f"Answer in a {tone_r} tone, {style_r} style. {right_q}", st.session_state.right_index, model_prompt_tuple)
            if cite_toggle_r:
                ev = st.session_state.get("LAST_EVIDENCE", [])
                cite_text = "\n\nSources:\n" + "\n".join([f"- Chunk {i+1}" for i, _ in enumerate(ev)]) if ev else ""
            else:
                cite_text = ""
            st.session_state["CHAT_RIGHT"].append({"role": "assistant", "content": (ans or "") + cite_text})
            with st.chat_message("assistant"):
                full = (ans or "") + cite_text
                chunks = [full[i:i+600] for i in range(0, len(full), 600)] if full else []
                ph = st.empty()
                acc = ""
                for ch in chunks:
                    acc += ch
                    ph.markdown(acc)
            sugg_cols_r = st.columns(3)
            followups_r = ["Explain further", "Give examples", "List key risks"]
            for i, f in enumerate(followups_r):
                if sugg_cols_r[i].button(f, key=f"rfu_{i}"):
                    fq = f"{f.lower()} related to: {right_q}"
                    ans2 = ask_with_index(f"Answer in a {tone_r} tone. {fq}", st.session_state.right_index, model_prompt_tuple)
                    st.session_state["CHAT_RIGHT"].append({"role": "assistant", "content": ans2})
                    st.experimental_rerun()
        regen_cols_r = st.columns(2)
        if regen_cols_r[0].button("🔄 Regenerate", key="right_regen") and st.session_state.get("CHAT_RIGHT"):
            last_user = next((m for m in reversed(st.session_state["CHAT_RIGHT"]) if m["role"]=="user"), None)
            if last_user:
                ans = ask_with_index(f"Answer in a {tone_r} tone. {last_user['content']}", st.session_state.right_index, model_prompt_tuple)
                st.session_state["CHAT_RIGHT"].append({"role": "assistant", "content": ans})
                st.experimental_rerun()
        refine_r = regen_cols_r[1].text_input("Refine instruction (Right)")
        if refine_r and st.button("✨ Apply Refinement", key="apply_refine_right") and st.session_state.get("CHAT_RIGHT"):
            last_assist = next((m for m in reversed(st.session_state["CHAT_RIGHT"]) if m["role"]=="assistant"), None)
            if last_assist:
                model, prompt = get_conversational_chain()
                refined = model.invoke(f"Refine the answer per: {refine_r}.\nOriginal Answer:\n{last_assist['content']}")
                st.session_state["CHAT_RIGHT"].append({"role": "assistant", "content": refined})
                st.experimental_rerun()
        note = st.text_area("Add note (Right)", key="note_right")
        if st.button("Save Note (Right)"):
            notes = st.session_state.get("NOTES_RIGHT", [])
            notes.append({"q": right_q, "a": (st.session_state["CHAT_RIGHT"][-1]["content"] if st.session_state.get("CHAT_RIGHT") else ""), "note": note})
            st.session_state["NOTES_RIGHT"] = notes
            st.success("Saved note.")
        tc_right = st.session_state.get("CHAT_RIGHT", [])
        if tc_right:
            if st.button("📤 Export Chat Transcript (Right)"):
                md = "\n\n".join([("**User:** " + m["content"]) if m["role"]=="user" else ("**Assistant:** " + m["content"]) for m in tc_right])
                st.download_button("Download Transcript", md.encode("utf-8"), file_name="right_chat.md")
            pin_idx_r = st.number_input("Pin answer # (Right)", min_value=1, max_value=len(tc_right), value=len(tc_right))
            if st.button("📌 Pin Answer (Right)", key="pin_right"):
                m = tc_right[pin_idx_r-1]
                st.session_state.setdefault("PINNED_RIGHT", []).append(m)
                st.success("Pinned.")
        themed_panel_end()
    st.write("---")
    themed_panel_start()
    st.subheader("🆚 Compare & Chat Across Sides")
    compare_q = st.text_input("Enter a question to ask both sides")
    if compare_q:
        mode = st.radio("Evidence mode", ["Single-doc", "Cross-doc fused"], horizontal=True)
        if mode == "Cross-doc fused":
            fused_ans = fused_evidence_answer(compare_q, st.session_state.left_index, st.session_state.right_index, model_prompt_tuple)
            st.markdown("**Fused Answer (Cross-doc)**")
            st.write(fused_ans)
        left_ans = ask_with_index(compare_q, st.session_state.left_index, model_prompt_tuple)
        right_ans = ask_with_index(compare_q, st.session_state.right_index, model_prompt_tuple)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Left Answer**")
            st.write(left_ans)
        with c2:
            st.markdown("**Right Answer**")
            st.write(right_ans)
        st.markdown("**Semantic Compare (heuristic)**")
        la = (left_ans or "").split()
        ra = (right_ans or "").split()
        left_only = [w for w in la if w not in ra][:50]
        right_only = [w for w in ra if w not in la][:50]
        st.write({"left_only": left_only, "right_only": right_only})
        # Confidence proxy: simple coverage based on evidence count
        left_cov = len(st.session_state.get("LAST_EVIDENCE", []))
        # Re-run right to fetch evidence info (stored last), so preserve separately
        st.session_state["LEFT_LAST_EVIDENCE"] = st.session_state.get("LAST_EVIDENCE", [])
        _ = ask_with_index(compare_q, st.session_state.right_index, model_prompt_tuple)
        st.session_state["RIGHT_LAST_EVIDENCE"] = st.session_state.get("LAST_EVIDENCE", [])
        right_cov = len(st.session_state.get("RIGHT_LAST_EVIDENCE", []))
        st.info({"left_evidence_chunks": left_cov, "right_evidence_chunks": right_cov})
        # Export report
        if st.button("Export Compare Report (Markdown)"):
            report = f"# Compare Report\n\n## Question\n{compare_q}\n\n## Left Answer\n{left_ans}\n\n## Right Answer\n{right_ans}\n\n## Differences\nLeft-only: {', '.join(left_only)}\n\nRight-only: {', '.join(right_only)}\n\n"
            st.download_button("Download Report", report, file_name="compare_report.md")
    themed_panel_end()
    st.write("---")
    themed_panel_start()
    st.subheader("📅 Interactive Timeline Builder")
    side_pick = st.selectbox("Choose side for timeline", ["Left", "Right"])
    side_text = st.session_state.left_text if side_pick == "Left" else st.session_state.right_text
    evts = extract_dates_events(side_text)
    if evts:
        for e in evts[:200]:
            st.write(f"{e['date']}: {e['text']}")
        if st.button("Export Timeline (CSV)"):
            csv_content = "date,text\n" + "\n".join([f"{e['date']},{e['text'].replace(',', ' ')}" for e in evts])
            st.download_button("Download CSV", csv_content, file_name="timeline.csv")
    else:
        st.info("No dates detected. Try other documents.")
    themed_panel_end()
    st.write("---")
    themed_panel_start()
    st.subheader("✅ Task & Action Extractor")
    plan_q = st.text_input("Describe your desired action plan focus (optional)", key="plan_q")
    if st.button("Generate Action Plan"):
        model, _ = get_conversational_chain()
        context = (st.session_state.left_text or "") + "\n" + (st.session_state.right_text or "")
        prompt = (
            "Extract actionable tasks with owners, due dates, and dependencies from the following context. "
            "Return as concise bullet points. "
            f"Focus: {plan_q}\n\nContext:\n{context[:8000]}"
        )
        action_plan = model.invoke(prompt)
        st.write(action_plan)
        if st.button("Export Plan (Markdown)"):
            st.download_button("Download Plan", action_plan or "", file_name="action_plan.md")
    themed_panel_end()
    st.write("---")
    themed_panel_start()
    st.subheader("📑 Policy/Contract Redline (Clause Diff)")
    if st.session_state.left_text and st.session_state.right_text:
        left_clauses = clause_split(st.session_state.left_text)
        right_clauses = clause_split(st.session_state.right_text)
        n = min(len(left_clauses), len(right_clauses), 20)
        for i in range(n):
            lc, rc = left_clauses[i], right_clauses[i]
            risk_l, risk_r = simple_risk_score(lc), simple_risk_score(rc)
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Left Clause {i+1} (risk {risk_l}/10)**")
                st.code(lc[:800])
            with cols[1]:
                st.markdown(f"**Right Clause {i+1} (risk {risk_r}/10)**")
                st.code(rc[:800])
    else:
        st.info("Upload and process both sides to view clause redlines.")
    themed_panel_end()

    with st.sidebar:
        st.title("🎨 Theme & Accents")
        st.session_state["ACCENT_THEME"] = st.selectbox(
            "Accent colors",
            ["Purple/Teal", "Pink/Orange", "Blue/Lime", "Red/Gold"],
            index=["Purple/Teal", "Pink/Orange", "Blue/Lime", "Red/Gold"].index(st.session_state.get("ACCENT_THEME", "Purple/Teal"))
        )
        st.session_state["THEME_STYLE"] = st.selectbox(
            "Design style",
            ["Neon", "Glass"],
            index=["Neon", "Glass"].index(st.session_state.get("THEME_STYLE", "Neon"))
        )
        st.caption("Pick your accent palette — colorful UI everywhere.")
        st.write("---")
        st.title("🔑 Google API Key")
        api_key_input = st.text_input("Enter Google API Key", type="password", placeholder="GOOGLE_API_KEY")
        if st.button("Save API Key"):
            if api_key_input and api_key_input.strip():
                st.session_state["GOOGLE_API_KEY"] = api_key_input.strip()
                ensure_api_key()
                st.success("API key saved and configured.")
            else:
                st.warning("Please enter a valid API key.")
        st.write("---")
        st.title("📦 Projects (Save/Load)")
        proj_name = st.text_input("Project name", value=st.session_state.get("PROJECT_NAME", ""))
        st.session_state["PROJECT_NAME"] = proj_name
        proj_dir = "docs/projects"
        os.makedirs(proj_dir, exist_ok=True)
        if st.button("Snapshot Project"):
            snapshot = {
                "timestamp": datetime.datetime.now().isoformat(),
                "name": proj_name or f"project-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "settings": {
                    "EMBEDDING_PROVIDER": st.session_state.get("EMBEDDING_PROVIDER"),
                    "LLM_PROVIDER": st.session_state.get("LLM_PROVIDER"),
                    "RETRIEVAL_K": st.session_state.get("RETRIEVAL_K"),
                    "RERANK_BY_HEADINGS": st.session_state.get("RERANK_BY_HEADINGS"),
                    "FOCUS_TERMS": st.session_state.get("FOCUS_TERMS"),
                    "CHUNK_SIZE": st.session_state.get("CHUNK_SIZE"),
                },
                "left_text": st.session_state.get("left_text", ""),
                "right_text": st.session_state.get("right_text", ""),
                "notes_left": st.session_state.get("NOTES_LEFT", []),
                "notes_right": st.session_state.get("NOTES_RIGHT", []),
            }
            path = os.path.join(proj_dir, f"{snapshot['name']}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                st.success(f"Snapshot saved: {path}")
            except Exception as e:
                st.error(f"Snapshot failed: {e}")
        # Restore
        try:
            files = [f for f in os.listdir(proj_dir) if f.endswith('.json')]
        except Exception:
            files = []
        pick = st.selectbox("Restore project", options=[""] + files)
        if pick and st.button("Restore"):
            try:
                with open(os.path.join(proj_dir, pick), "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.session_state["PROJECT_NAME"] = data.get("name", "")
                for k, v in data.get("settings", {}).items():
                    st.session_state[k] = v
                st.session_state["left_text"] = data.get("left_text", "")
                st.session_state["right_text"] = data.get("right_text", "")
                st.session_state["NOTES_LEFT"] = data.get("notes_left", [])
                st.session_state["NOTES_RIGHT"] = data.get("notes_right", [])
                st.success("Project restored. Re-process to rebuild indexes.")
            except Exception as e:
                st.error(f"Restore failed: {e}")
        st.caption("Snapshots store text and notes; re-index after restore.")
        st.write("---")
        st.title("🔑 OpenAI API Key")
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", placeholder="OPENAI_API_KEY")
        if st.button("Save OpenAI Key"):
            if openai_key_input and openai_key_input.strip():
                st.session_state["OPENAI_API_KEY"] = openai_key_input.strip()
                os.environ["OPENAI_API_KEY"] = openai_key_input.strip()
                st.success("OpenAI key saved.")
            else:
                st.warning("Please enter a valid OpenAI API key.")
        st.write("---")
        st.title("📐 Embedding Provider")
        st.session_state["EMBEDDING_PROVIDER"] = st.selectbox(
            "Choose provider for embeddings",
            ["OpenAI", "Google Gemini", "Local (No-Quota)"]
        )
        st.caption("Use Local if you hit quotas or 429 errors.")
        st.write("---")
        st.title("🔎 Retrieval Settings")
        st.session_state["RETRIEVAL_K"] = st.slider("Top-k chunks", min_value=2, max_value=10, value=st.session_state.get("RETRIEVAL_K", 4))
        st.caption("Increase k for broader context; lower for precision.")
        st.write("---")
        st.title("🧭 Advanced RAG Controls")
        st.session_state["RERANK_BY_HEADINGS"] = st.checkbox("Rerank by section headings", value=st.session_state.get("RERANK_BY_HEADINGS", False))
        st.session_state["FOCUS_TERMS"] = st.text_input("Focus terms (comma-separated)", value=st.session_state.get("FOCUS_TERMS", ""))
        st.session_state["CHUNK_SIZE"] = st.slider("Chunk size (auto-tune)", min_value=2000, max_value=60000, value=st.session_state.get("CHUNK_SIZE", 50000), step=1000)
        st.caption("Adjust chunking for better recall/precision; rerank emphasizes sections.")
        if st.button("Optimize Retrieval"):
            # Simple heuristic: if k<4 and many differences, increase k; else reduce.
            k = st.session_state.get("RETRIEVAL_K", 4)
            st.session_state["RETRIEVAL_K"] = min(10, k+1)
            st.success("Retrieval tuned: increased top-k.")
        st.title("🧠 LLM Provider")
        st.session_state["LLM_PROVIDER"] = st.selectbox(
            "Choose provider for chat",
            ["OpenAI", "Google Gemini"]
        )
        st.caption("OpenAI uses gpt-4o-mini (low tokens, high quality).")
        st.write("---")

        st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 (Optional) Single-Panel Upload")
        pdf_docs = st.file_uploader("Drag & drop or browse PDFs (Single)", accept_multiple_files=True, type=["pdf"], key="single_upl")
        if st.button("Process Single", key="process_single"):
            with st.spinner("Indexing single panel documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                idx = build_vector_store(text_chunks)
                st.session_state.left_index = idx
                st.session_state.left_text = raw_text
                st.success("Single panel documents indexed (Left).")
        
        st.write("---")
        st.image("img/pic.jpg")
        st.write("AI App created by @ Abhishek Kumar")  # add this line to display the image


    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/abhishekkumar62000" target="_blank">Abhishek Kumar Yadav</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

    # Live PDF Region Chat and Table Extraction
    st.write("---")
    themed_panel_start()
    st.subheader("🔍 Live PDF Region Chat")
    side_sel = st.selectbox("Side for region chat", ["Left", "Right"], key="region_side")
    pdf_files = left_files if side_sel == "Left" else right_files
    if pdf_files:
        try:
            # Let user pick a PDF by filename
            names = [getattr(f, 'name', f"doc-{i}") for i, f in enumerate(pdf_files)]
            which = st.selectbox("Pick PDF", options=list(range(len(names))), format_func=lambda i: names[i], key="region_pdf_pick")
            target = pdf_files[which]
            page_no = st.number_input("Page number (1-based)", min_value=1, value=1)
            query_text = st.text_input("Text filter in page (optional)")
            do_table = st.checkbox("Extract tables to CSV", value=False)
            if st.button("Analyze Page Region"):
                with pdfplumber.open(target) as pdf:
                    pidx = max(0, min(page_no-1, len(pdf.pages)-1))
                    page = pdf.pages[pidx]
                    page_text = page.extract_text() or ""
                    selection = page_text
                    if query_text.strip():
                        # Keep lines containing query_text
                        lines = [ln for ln in (page_text or "").splitlines() if query_text.lower() in ln.lower()]
                        selection = "\n".join(lines) or page_text
                    model, prompt = get_conversational_chain()
                    q = st.text_input("Ask about this selection", key="region_q")
                    if q:
                        formatted = prompt.format(context=selection, question=q)
                        ans = model.invoke(formatted)
                        st.write(ans)
                    if do_table:
                        tables = page.extract_tables()
                        if tables:
                            for ti, tbl in enumerate(tables[:5]):
                                df = pd.DataFrame(tbl[1:], columns=tbl[0]) if tbl and len(tbl) > 1 else pd.DataFrame(tbl)
                                st.dataframe(df)
                                csv_bytes = df.to_csv(index=False).encode("utf-8")
                                st.download_button(f"Download Table {ti+1} (CSV)", csv_bytes, file_name=f"page{pidx+1}_table{ti+1}.csv")
                        else:
                            st.info("No tables detected on this page.")
        except Exception as e:
            st.warning(f"Region chat encountered an issue: {e}")
    themed_panel_end()

    # Multi-Modal Compare: table-aware diff (basic)
    st.write("---")
    themed_panel_start()
    st.subheader("📊 Multi-Modal Compare (Tables)")
    if left_files and right_files:
        try:
            # Select PDFs and page to compare tables quickly
            ln = [getattr(f, 'name', f"left-{i}") for i, f in enumerate(left_files)]
            rn = [getattr(f, 'name', f"right-{i}") for i, f in enumerate(right_files)]
            li = st.selectbox("Left PDF", options=list(range(len(ln))), format_func=lambda i: ln[i], key="mmc_left_pdf")
            ri = st.selectbox("Right PDF", options=list(range(len(rn))), format_func=lambda i: rn[i], key="mmc_right_pdf")
            page_cmp = st.number_input("Page (1-based)", min_value=1, value=1)
            left_tbls, right_tbls = [], []
            with pdfplumber.open(left_files[li]) as lpdf:
                lp = lpdf.pages[max(0, min(page_cmp-1, len(lpdf.pages)-1))]
                left_tbls = lp.extract_tables() or []
            with pdfplumber.open(right_files[ri]) as rpdf:
                rp = rpdf.pages[max(0, min(page_cmp-1, len(rpdf.pages)-1))]
                right_tbls = rp.extract_tables() or []
            if left_tbls and right_tbls:
                ldf = pd.DataFrame(left_tbls[0][1:], columns=left_tbls[0][0]) if len(left_tbls[0]) > 1 else pd.DataFrame(left_tbls[0])
                rdf = pd.DataFrame(right_tbls[0][1:], columns=right_tbls[0][0]) if len(right_tbls[0]) > 1 else pd.DataFrame(right_tbls[0])
                st.write("Left Table")
                st.dataframe(ldf)
                st.write("Right Table")
                st.dataframe(rdf)
                # Harmonize on common columns
                common = [c for c in ldf.columns if c in set(rdf.columns)]
                if common:
                    lsub, rsub = ldf[common].copy(), rdf[common].copy()
                    # Basic discrepancy summary: mismatched counts and value diffs
                    diff_rows = []
                    max_rows = max(len(lsub), len(rsub))
                    for i in range(max_rows):
                        lv = lsub.iloc[i].to_dict() if i < len(lsub) else {c: None for c in common}
                        rv = rsub.iloc[i].to_dict() if i < len(rsub) else {c: None for c in common}
                        if lv != rv:
                            diff_rows.append({"row": i+1, "left": lv, "right": rv})
                    st.write({"rows_compared": max_rows, "differences": len(diff_rows)})
                    if diff_rows:
                        diff_csv = "row," + ",".join([f"left_{c}" for c in common]) + "," + ",".join([f"right_{c}" for c in common]) + "\n"
                        for d in diff_rows:
                            lc = ",".join([str(d["left"].get(c, "")) for c in common])
                            rc = ",".join([str(d["right"].get(c, "")) for c in common])
                            diff_csv += f"{d['row']},{lc},{rc}\n"
                        st.download_button("Download Table Diff (CSV)", diff_csv.encode("utf-8"), file_name="table_diff.csv")
                else:
                    st.info("No common columns to compare.")
            else:
                st.info("No tables detected on one or both pages.")
        except Exception as e:
            st.warning(f"Multi-modal compare issue: {e}")
    themed_panel_end()

    # Policy Risk Studio (playbooks)
    st.write("---")
    themed_panel_start()
    st.subheader("🛡️ Policy Risk Studio")
    playbook = st.selectbox("Choose playbook", ["HR Policy", "SaaS Contract", "Compliance Checklist"], key="pbs_pick")
    if st.button("Analyze with Playbook"):
        base = (st.session_state.left_text or "") + "\n" + (st.session_state.right_text or "")
        tags = {
            "HR Policy": ["leave", "harassment", "benefits", "termination"],
            "SaaS Contract": ["sla", "uptime", "liability", "security", "data", "termination"],
            "Compliance Checklist": ["gdpr", "hipaa", "pci", "audit", "retention"],
        }[playbook]
        # Tag clauses by keywords
        clauses = clause_split(base)
        tagged = []
        for cl in clauses[:50]:
            found = [t for t in tags if t in cl.lower()]
            risk = simple_risk_score(cl) + len(found)
            tagged.append({"clause": cl[:400], "tags": found, "risk": min(10, risk)})
        st.write(tagged)
        # Suggest remediation via LLM
        model, _ = get_conversational_chain()
        prompt = (
            f"Given these tagged clauses and risks for {playbook}, suggest remediation text and produce a redlined draft with tracked changes markers.\n\n"
            f"Clauses:\n{json.dumps(tagged, ensure_ascii=False)}\n"
        )
        redlined = model.invoke(prompt)
        st.write(redlined)
        st.download_button("Download Redlined Draft (Markdown)", (redlined or "").encode("utf-8"), file_name="redlined_draft.md")
    themed_panel_end()

    # Team Collaboration & Approvals
    st.write("---")
    themed_panel_start()
    st.subheader("👥 Team Collab + Review Loops")
    reviewer = st.text_input("Reviewer name")
    review_note = st.text_area("Annotation / comment")
    due_date = st.text_input("Due date (YYYY-MM-DD)")
    approve = st.checkbox("Approve this answer/evidence")
    if st.button("Add Review Record"):
        ledger = st.session_state.get("APPROVAL_LEDGER", [])
        ledger.append({
            "ts": datetime.datetime.now().isoformat(),
            "reviewer": reviewer,
            "note": review_note,
            "due": due_date,
            "approved": approve,
        })
        st.session_state["APPROVAL_LEDGER"] = ledger
        st.success("Review record added.")
    if st.session_state.get("APPROVAL_LEDGER"):
        st.write(st.session_state.get("APPROVAL_LEDGER"))
        # Export approval ledger
        rows = st.session_state.get("APPROVAL_LEDGER", [])
        if rows:
            keys = ["ts", "reviewer", "note", "due", "approved"]
            csv_out = ",".join(keys) + "\n"
            for r in rows:
                csv_out += ",".join([str(r.get(k, "")) for k in keys]) + "\n"
            st.download_button("Download Approval Ledger (CSV)", csv_out.encode("utf-8"), file_name="approval_ledger.csv")
    themed_panel_end()

if __name__ == "__main__":
    main()

