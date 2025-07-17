# Indian Legal AI Assistant

## Overview
This project develops a Retrieval-Augmented Generation (RAG)-based AI assistant for Indian law, designed to assist lawyers and laymen. It leverages Indian Kanoon data to provide legal research, case summaries, and court notice analysis, with plans for international precedent integration and a user-friendly interface.

## Features
- **Lawyer Support**: Answers queries like "Latest Supreme Court judgments on Section 138 NI Act" with case summaries and legal explanations.
- **Layman Support**: Assists with personal issues (e.g., "My landlord isn’t returning my deposit") and analyzes scanned court notices.
- **Data Pipeline**: Scrapes and indexes Indian Kanoon cases using API, with dynamic query updates.
- **Technologies**: LangChain, ChromaDB, SentenceTransformers, OpenAI API, OCR (pytesseract).

## Current Progress (July 17, 2025)
- **Completed**:
  - Built `ik_api_scraper.py` to fetch and save 60 cases (e.g., IPC 302, Section 138 NI Act) with query-based naming.
  - Implemented `update_dataset.py` for dataset management and indexing in ChromaDB.
  - Developed `rag_chain.py` for RAG with OCR support, handling lawyer and layman queries, and caching to optimize API usage.
  - Resolved initial 429 errors (token limit) with context truncation and model optimization.
- **In Progress**:
  - Testing `rag_chain.py` for robustness and opponent argument prediction.
  - Adding automation scheduling for dataset updates.
- **Next Steps**:
  - Enhance notice analysis with opponent arguments.
  - Integrate international precedents (US/EU cases).
  - Develop a Streamlit UI for deployment.

## Installation
1. Clone the repository:
   
   git clone https://github.com/kharishgit/legal-ai-assistant.git
   cd legal-ai-assistant

2. Set up a virtual environment:
    python -m venv legal-bot
    source legal-bot/bin/activate

3. Install dependencies:
    pip install -r requirements.txt

4. Run the RAG
    python src/rag/rag_chain.py

## Contributing
    Contributions are welcome! Please fork the repository and submit pull requests.