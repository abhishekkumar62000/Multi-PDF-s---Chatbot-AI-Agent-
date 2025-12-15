<img width="1920" height="1080" alt="Page 4" src="https://github.com/user-attachments/assets/0cb3543c-f8b8-44a7-a8e9-52424a54c4de" />

<img width="1916" height="1080" alt="Page1" src="https://github.com/user-attachments/assets/3eec8d87-d7a8-4118-8f8f-b38c9371127f" />


**App Demo:**-

https://github.com/user-attachments/assets/c310920a-bd84-4336-a3f1-eb5abdc3af04

---

# 📚 Multi-PDFs – Chat Agent 🤖

### Discover, Chat & Compare Insights Across Your Documents

🚀 **Live App:**
👉 [https://multi-pdf-chat-ai-agent.streamlit.app/](https://multi-pdf-chat-ai-agent.streamlit.app/)

---

## 🧠 Overview

**Multi-PDFs – Chat Agent** is an advanced **multi-document RAG (Retrieval-Augmented Generation) chatbot** built using **Streamlit**, **FAISS**, and **LLMs (Google Gemini / OpenAI)**.

The app allows users to **upload multiple PDFs**, chat with them intelligently, **compare documents side-by-side**, extract insights like **timelines, clauses, tables**, and generate **action plans, risk analyses, and summaries** — all inside a modern, themed UI.

This tool is designed for:

* 📄 Researchers
* ⚖️ Legal & policy analysts
* 🏢 Business & compliance teams
* 🎓 Students & educators
* 🤖 AI/ML & RAG enthusiasts

---

## ✨ Key Highlights

* 🔍 **Multi-Document RAG Chat**
* ⚖️ **Side-by-Side PDF Comparison**
* 🧩 **Evidence-backed answers with citations**
* 🧠 **Gemini & OpenAI LLM support**
* 🗂️ **Project snapshots & collaboration tools**
* 🎨 **Modern Neon / Glass UI themes**

---

## 🧩 Core Features

### 🔁 Dual-Panel RAG Chat

* Upload and process PDFs separately on **Left** and **Right** panels
* Independent chat history per panel
* Regenerate, refine, pin, and export answers
* Session-based memory for smooth workflows

---

### 📑 Evidence & Citations

* Optional citation mode shows **exact chunk references**
* Stores last-used evidence per panel
* Improves trust, traceability, and accuracy

---

### ⚖️ Cross-Document Comparison

* Ask **one question** → get **Left vs Right answers**
* Heuristic **semantic diff** highlights differences
* Export full comparison as **Markdown report**

---

### 🔀 Fused Evidence Mode

* Combines top-k chunks from both documents
* Produces a **single unified answer**
* Ideal for synthesis and final conclusions

---

### 🗓️ Timeline Extraction

* Auto-detects date patterns from documents
* Generates structured timelines
* Export results as **CSV**

---

### 📋 Action Plan Generator

* Uses LLM to extract:

  * Tasks
  * Owners
  * Deadlines
  * Dependencies
* Exportable as **Markdown**

---

### 📜 Clause Redline & Risk Diff

* Splits text into clauses (by headings)
* Side-by-side clause comparison
* Simple **risk scoring** for policy/legal review

---

### 📍 Live PDF Region Chat

* Select **PDF + page number**
* Chat with page-specific content
* Optional table extraction
* Download extracted tables as **CSV**

---

### 📊 Multi-Modal Table Comparison

* Extract tables from selected pages on both sides
* Show dataframes inside UI
* Compute simple row-level diffs
* Export tables as CSV

---

### ⚠️ Policy Risk Studio

* Pre-built playbooks:

  * HR Policy
  * SaaS Contracts
  * Compliance Docs
* Keyword-based clause tagging
* Risk scoring + remediation suggestions
* Generate **redlined drafts using LLM**

---

### 👥 Team Collaboration Ledger

* Add reviewers, notes, due dates
* Approval tracking
* Export approvals ledger to CSV

---

### 💾 Project Snapshots

* Save entire project state:

  * Settings
  * Text
  * Notes
  * Indexes
* Restore anytime
* Enables long-running research workflows

---

### 📥 Single-Panel Quick Upload

* Sidebar upload to directly process docs into **Left panel**
* Faster for solo analysis

---

### 🎨 Theming & UI

* Neon / Glass / Accent palette themes
* Custom hero sections and panels
* Implemented via `ui_theme.py`

---

### 🔌 Provider Flexibility

* **Embeddings**

  * Google (embedding-001)
  * OpenAI (text-embedding-3-small)
  * Local hash embeddings (no quota)
* **LLMs**

  * Google Gemini (1.5 Flash)
  * OpenAI (gpt-4o-mini)

---

### ⚙️ Advanced RAG Controls

* Chunk size & overlap tuning
* Top-K retrieval control
* Heading-aware reranking
* Focus terms boosting
* “Optimize Retrieval” quick button

---

### 📤 Transcript Export & Pinning

* Export chats as Markdown
* Pin important answers for later reference

---

## 🧪 `chatapp.py` – Extra Capabilities

* 📄 TXT & DOCX uploads
* 🔎 Semantic search across all documents
* 🌐 Language selector
* 🧠 Knowledge graph (entities & tags)
* 🧾 Page-wise summarization
* 📊 Usage analytics (most asked, coverage)
* 📥 Export chat history to TXT/PDF

---

## 🛠️ How It Works (Architecture)

### 1️⃣ Upload & Processing

* PDFs loaded using **PyPDF2**
* Text chunked using `RecursiveCharacterTextSplitter`

  * Default: 50k chunk size, 1k overlap (configurable)

---

### 2️⃣ Embeddings

* Provider options:

  * **Google:** `models/embedding-001`
  * **OpenAI:** `text-embedding-3-small`
  * **Local:** `SimpleHashEmbeddings`
* Indexed using **FAISS**
* `chatapp.py` supports persistent FAISS index

---

### 3️⃣ Retrieval & Chat

* Top-K similarity search
* Optional focus-term boosting
* Strict prompt template:

  * Context-only answers
  * No hallucinations
* Follow-up suggestions + refine options

---

### 4️⃣ LLM Providers

* **Google Gemini:** `gemini-1.5-flash-latest`
* **OpenAI:** `gpt-4o-mini`

---

### 5️⃣ Comparisons & Analytics

* Fused cross-doc mode
* Semantic answer diff
* Clause risk detection
* Timeline CSV export
* Page-level & table-level tools

---

### 6️⃣ State & Projects

* Managed via Streamlit `session_state`
* Project snapshots stored as JSON
* Restore triggers re-indexing automatically

---

## 🚀 Quick Start (User Flow)

1. Upload PDFs to **Left** and/or **Right**
2. Click **Process** → FAISS indexes created
3. Ask questions in each panel
4. Compare answers or fuse evidence
5. Extract timelines, tables, clauses, or action plans
6. Export reports or save project snapshots

---

## 🔐 Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

(Set via Streamlit Cloud → Secrets)

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **Vector Store:** FAISS
* **LLMs:** Google Gemini, OpenAI
* **Embeddings:** Google / OpenAI / Local
* **PDF Parsing:** PyPDF2, pdfplumber
* **Language:** Python

---

## 🌟 Why This Project?

This project demonstrates:

* Real-world **RAG architecture**
* Multi-document reasoning
* Advanced UX for AI apps
* Production-ready Streamlit patterns
* Legal, policy & research use-cases

---

---

# 🌳 3️⃣ LangGraph-Style Binary Tree

### “How a New User Uses the App”

> **Binary Tree format (Decision-based user flow)**

```text
                             [ New User Opens App ]
                                      |
                     -----------------------------------
                     |                                 |
             [ Upload Documents? ]                [ View Demo Only ]
                     |
         ---------------------------------
         |                               |
   [ Single PDF ]                  [ Multiple PDFs ]
         |                               |
   [ Left Panel ]             -------------------------
                               |                       |
                         [ Left Panel ]         [ Right Panel ]
                               |                       |
                        [ Click Process ]        [ Click Process ]
                               |                       |
                    [ FAISS Index Created ]   [ FAISS Index Created ]
                               |                       |
                    -----------------------------------------
                    |                                       |
            [ Ask Individual Questions ]            [ Compare Mode ]
                    |                                       |
        -----------------------------           ----------------------------
        |                           |           |                          |
 [ View Answer ]           [ Enable Citations ]  [ Left vs Right Answer ]  [ Fused Answer ]
        |                           |           |                          |
 [ Regenerate / Refine ]     [ Evidence Stored ] [ Semantic Diff ]      [ Cross-Doc RAG ]
        |                           |           |                          |
 [ Pin / Export Chat ]       [ Export Evidence ] [ Export Compare MD ]  [ Unified Insight ]
        |
 ----------------------------
 |                          |
[ Advanced Tools ]     [ Save Project ]
 |                          |
 |              -----------------------------
 |              |                           |
 |        [ Timeline / Tables ]       [ Project Snapshot ]
 |              |                           |
 |        [ CSV Export ]          [ Restore Later Session ]
 |
[ Policy / Clause / Risk Studio ]
 |
[ Final Analysis & Reports ]
```

---

# 🧠 4️⃣ LangGraph-Inspired Node Explanation (Conceptual)

| Node              | Description                         |
| ----------------- | ----------------------------------- |
| **InputNode**     | User uploads PDFs / DOCX / TXT      |
| **ParserNode**    | Extracts text (PyPDF2 / pdfplumber) |
| **ChunkNode**     | Splits text into chunks             |
| **EmbeddingNode** | Generates embeddings                |
| **VectorNode**    | Stores embeddings in FAISS          |
| **RetrieverNode** | Fetches Top-K similar chunks        |
| **PromptNode**    | Injects context into prompt         |
| **LLMNode**       | Gemini / OpenAI generates answer    |
| **CompareNode**   | Left vs Right / Fused logic         |
| **AnalyticsNode** | Timeline, tables, clauses           |
| **OutputNode**    | UI render + export options          |

This mirrors **LangGraph DAG logic**, even though Streamlit is event-driven.

---

## 🙌 Author

**Abhishek Kumar (Abhi Yadav)**
AI Engineer | Data Science Aspirant | Builder
💡 Passionate about RAG systems, Agentic AI & real-world AI products

---

## ❤️ **Made with Passion by Abhishek Yadav & Open-Source Contributors!** 🚀✨


<h1 align="center">© LICENSE <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Telegram-Animated-Emojis/main/Symbols/Check%20Box%20With%20Check.webp" alt="Check Box With Check" width="25" height="25" /></h1>

<table align="center">
  <tr>
     <td>
       <p align="center"> <img src="https://github.com/malivinayak/malivinayak/blob/main/LICENSE-Logo/MIT.png?raw=true" width="80%"></img>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/> <br> 
This project is licensed under <a href="./LICENSE">MIT</a>. <img width=2300/>
    </td>
  </tr>
</table>

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900">



 
 <hr>

<div align="center">
<a href="#"><img src="assets/githubgif.gif" width="150"></a>
	
### **Thanks for checking out my GitHub Profile!**  

 ## 💌 Sponser

  [![BuyMeACoffee](https://img.buymeacoffee.com/button-api/?text=Buymeacoffee&emoji=&slug=codingstella&button_colour=FFDD00&font_colour=000000&font_family=Comic&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/abhishekkumar62000)

## 👨‍💻 Developer Information  
**Created by:** **Abhishek Kumar**  
**📧 Email:** [abhiydv23096@gmail.com](mailto:abhiydv23096@gmail.com)  
**🔗 LinkedIn:** [Abhishek Kumar](https://www.linkedin.com/in/abhishek-kumar-70a69829a/)  
**🐙 GitHub Profile:** [@abhishekkumar62000](https://github.com/abhishekkumar62000)

<p align="center">
  <img src="https://github.com/user-attachments/assets/6283838c-8640-4f22-87d4-6d4bfcbbb093" width="120" style="border-radius: 50%;">
</p>
</div>  


`Don't forget to give A star to this repository ⭐`


`👍🏻 All Set! 💌`

</div>

---

