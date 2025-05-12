# Technical Exercise: RAG Chat Tool with Chunkr API

**Time allotment:** 24 hours  
**Language(s):** Rust, Python or TypeScript  
**Modalities accepted:** CLI, API server, React app, Streamlit app, etc.

---

## 📖 Background

In this exercise, you'll build a simple RAG chat application using the Chunkr API to ingest and query a single document (provided).

We'll supply you with:
- An **OpenRouter API key** (for LLM calls)
- A **Chunkr API key** (for document ingestion)

---

## 🎯 Goal

Build a user-facing tool that:
1. Ingests one provided document into Chunkr.
2. Allows a user to chat with the content.
3. Returns answers with **citations** (page- or chunk-level).

---

## 📦 Requirements

1. **Ingestion**  
   - Upload the provided file to Chunkr.

2. **Retrieval & Chat**  
   - Implement a query pipeline that:
     - Retrieves relevant context from Chunkr.
     - Calls an LLM (via OpenRouter).
     - Returns an answer with inline citations to the original document.

3. **README**  
   - Setup & install steps.
   - How to run & test the tool.
   - High-level design & feature list.

4. **Code Quality**  
   - Clear project structure.
   - Meaningful naming, comments, and error handling.
   - Use of models and types (eg. pydantic)

Your solution could be a CLI, API server, Full stack implementation, etc. 

---

## 💡 Tips & Guidelines

- **Context window**: You can retrieve top-k chunks with search or put entire pdf in context.
- **Tech stack**: Feel free to combine any frameworks (e.g., React + FastAPI) or keep it simple (Streamlit).
- **LLM choice**: Any model supported by OpenRouter.
- **UV** preferred if using python.

---

## 🎁 Bonus / Brownie Points

- Polished **UI/UX**
- **Deployment** (Heroku/Vercel/AWS/GCP).
- **Multi-file** ingestion & chat support.
- **CI/CD**: GitHub Actions workflows, linting, formatting, tests.
- **GitHub Releases** or version tagging.
- **Cookbook, blogs** or other forms of writing to express your code.
- **Docker Compose** for local deployment.
- **Search Layer** integrating Pinecone or other vector DBs.
> Any other creative feature or technology, just cook basically.

---

## 📬 Deliverables

1. A private Git repository shared with:
    - akhileshsharma99
    - ...
    - ...
2. A complete `README.md` (see Requirements).
3. Source code and configuration (e.g., `package.json`, `pyproject.toml`, `cargo.toml`) via GitHub.

Please submit a link to your repo within 24 hours.

---

## 🧪 Evaluation Criteria

- **Feature completeness:** Core RAG chat works end-to-end.
- **Creativity:** Unique ideas or extra features.
- **Code quality:** Readability, maintainability, tests.
- **Documentation:** Clarity of setup and design in README.
- **Bonus:** UI polish, deployment, CI, multi-file, etc.

Good luck, and we look forward to seeing your creativity shine! 🚀
