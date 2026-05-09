# DocuMind - Multiple PDF RAG Analyzer

A Streamlit app for searching and answering questions from multiple PDF documents using LangChain, FAISS, and Google Gemini.

## What it does

- Upload one or more PDFs
- Create a semantic search index
- Ask questions in natural language
- View answers with context from the documents

## Quick start

1. Clone the repository
   ```bash
   git clone https://github.com/akhilasuresh02/DocuMind---Multiple-PDF-Rag-Analyzer.git
   cd DocuMind
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env`
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```
5. Run the app
   ```bash
   streamlit run app.py
   ```

## Usage

- Upload PDF files
- Enter a question
- Read the AI-generated response
- Export conversation if needed

## Files

- `app.py` — main app
- `requirements.txt` — dependencies
- `.gitignore` — ignored files
- `README.md` — this file
- `faiss_index/` — generated index files

## Notes

- Keep `.env` private
- `faiss_index/` is generated automatically
- Use Python 3.10+
