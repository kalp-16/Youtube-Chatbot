# YouTube Chatbot with RAG and LLMs

A Streamlit-based chatbot that allows you to ask questions about YouTube videos. It fetches video transcripts, splits them into chunks, embeds them for retrieval, and uses a HuggingFace embeddings to answer questions in context.

---

## Features

- Fetches YouTube video transcripts (auto-generated or manual if available)
- Splits transcript into manageable chunks
- Stores embeddings in FAISS vector store for fast retrieval
- Uses LLM (Google Gemini or HuggingFace) for question answering
- Streamlit UI for an interactive chatbot experience
- Optional session memory to maintain conversation context

---
