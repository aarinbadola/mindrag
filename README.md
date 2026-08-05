---
title: MindRAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# MindRAG

RAG-based document intelligence system over a fixed set of up to 10 PDF documents.

Supports three query types:

- **Document QA** — factual questions answered from document content, with inline citations
- **Summarization** — map-reduce summarization of a single document or the full knowledge base
- **Conversation Recap** — a summary of the current chat session

Built with Gradio, Gemini 2.5 Flash, ChromaDB, and a cross-encoder reranker.

## Local development

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with `GEMINI_API_KEY=your-key-here` and `GROQ_API_KEY=your-key-here`
3. Drop your source PDFs into `docs/`
4. Run `python app.py`
