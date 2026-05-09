
# 📚 DocuMind - PDF RAG Chatbot

**A powerful Retrieval-Augmented Generation (RAG) application for intelligent Q&A across multiple PDF documents using Google Gemini AI and LangChain.**

> Chat with your PDFs, extract insights, and get AI-powered answers instantly.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-brightgreen?style=flat-square)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## ✨ Features

- 📄 **Multi-PDF Upload**: Upload multiple PDF files simultaneously
- 🤖 **AI-Powered Answers**: Uses Google Gemini 1.5 Flash for intelligent, contextual responses
- 🔍 **Semantic Search**: LangChain + FAISS for fast, accurate document retrieval
- ⚡ **Performance Optimized**: 
  - Parallel PDF extraction with ThreadPoolExecutor
  - Cached embeddings and FAISS indexes
  - BM25 + vector search hybrid retrieval
- 💬 **Chat Interface**: Conversational UI with user/bot avatars
- 📊 **Export Results**: Download conversation history as CSV
- 📈 **Ideal for**: Financial reports, annual statements, research papers, contracts, and more
- 🔐 **Secure**: API key stored in `.env` (not committed to git)

---

## 🎯 Use Cases

- 📋 Analyze **financial reports** and annual statements
- 📑 Extract insights from **research papers** and documentation
- 💼 Review **contracts** and legal documents
- 📊 Query **business data** across multiple documents
- 🔍 Extract **specific information** without reading entire PDFs

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** ([Get it here](https://ai.google.dev/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/documind.git
   cd documind
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file in the project root
   echo GOOGLE_API_KEY=your_api_key_here > .env
   ```
   
   Or create `.env` manually:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

---

## 💻 Usage

1. **Upload PDFs**: Click "Browse files" and select one or more PDF documents
2. **Ask Questions**: Type your question in the chat box
3. **Get Answers**: Receive AI-powered responses with relevant excerpts from your documents
4. **Export Chat**: Download your conversation history as a CSV file

### Example Queries

- "What are the main findings in this report?"
- "Summarize the financial performance"
- "What are the key risks mentioned?"
- "Compare these documents"
- "Extract all financial figures"

---

## 🏗️ Project Structure

```
documind/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── README.md             # This file
└── faiss_index/          # FAISS vector index (generated at runtime)
    └── index.faiss
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.43+ | Web interface |
| `langchain` | 0.3+ | RAG framework |
| `langchain-community` | 0.3+ | Community integrations |
| `langchain-google-genai` | 2.1+ | Google Gemini integration |
| `faiss-cpu` | 1.10+ | Vector search |
| `pypdf` / `PyPDF2` | Latest | PDF reading |
| `pandas` | 2.2+ | Data handling |
| `python-dotenv` | 1.0+ | Environment management |

See `requirements.txt` for the complete list with pinned versions.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Getting your Google Gemini API Key:**
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Click "Get API Key"
3. Create a new API key
4. Copy and paste it in `.env`

### Streamlit Configuration (Optional)

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#1F1F1F"

[client]
showErrorDetails = true

[logger]
level = "info"
```

---

## 🔧 Performance Optimizations

This application includes several optimizations for speed:

- **Parallel PDF Processing**: Uses ThreadPoolExecutor to extract multiple PDFs simultaneously
- **Cached Embeddings**: Embedding models are cached using Streamlit's `@st.cache_resource`
- **FAISS Index Caching**: Vector indexes are cached in session state to avoid reloading
- **BM25 Hybrid Search**: Combines semantic and keyword search for better retrieval
- **Batch Processing**: Optimized FAISS batch sizes for fewer merge passes

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `GOOGLE_API_KEY not found` | Create `.env` file with your API key |
| `PdfReadError` on some PDFs | Some malformed PDFs may fail; the app skips bad pages automatically |
| `FAISS index not found` | Upload PDFs first to generate the index |
| Slow performance | Increase `FAISS_BATCH_SIZE` in app.py or reduce document size |

### Enable Debug Logging

Check logs in the terminal running Streamlit for detailed debugging information.

---

## 📋 API Usage

The app uses the following APIs:

- **Google Gemini 1.5 Flash**: Text generation and Q&A
- **BAAI/bge-small-en-v1.5**: Embedding model (runs locally via FastEmbed)

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support & Questions

- **Issues**: Open an issue on GitHub
- **Discussions**: Start a discussion for questions
- **Email**: [your-email@example.com]

---

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/docs)
- [FAISS Documentation](https://faiss.ai/)
- [RAG Basics](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

---

## 📊 Performance Metrics

- **PDF Processing**: ~1-2 seconds per PDF (depending on size)
- **Query Response**: ~2-5 seconds (includes embedding + retrieval + generation)
- **Supported Document Size**: Up to 100+ MB (typical use case)

---

## ✅ Future Enhancements

- [ ] Support for additional document types (DOCX, TXT, images)
- [ ] Multi-language support
- [ ] Advanced filtering and search options
- [ ] Document summarization
- [ ] Citation tracking
- [ ] Integration with cloud storage (Google Drive, OneDrive)
- [ ] Custom model selection

---

**Made with ❤️ for document intelligence**

Last updated: May 2026

---

## 🔐 Google AI API Key

To use Gemini models and embeddings:

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Generate your API key
3. Enter the key in the **Streamlit sidebar**

---

## 📦 Tech Stack

| Tech       | Purpose                                  |
| ---------- | ---------------------------------------- |
| Streamlit  | UI framework for interactive web apps    |
| LangChain  | Managing LLM chains and embeddings       |
| Gemini 1.5 | Large Language Model (via Google AI API) |
| PyPDF2     | PDF text extraction                      |
| FAISS      | Vector database for similarity search    |
| Pandas     | Exporting conversation as CSV            |
| HTML/CSS   | Custom chat UI inside Streamlit          |

---

## 📁 File Structure

```
├── app.py               # Main Streamlit app
├── faiss_index/         # Folder where vectorstore is saved
├── requirements.txt     # Required Python packages
└── README.md            # You're here!
```

---

## 🧠 Prompt Template Logic

This tool is **finance-aware**. The prompt guides the LLM to:

* Evaluate financial statements from PDFs
* Detect irregularities or red flags
* Analyze related party transactions
* Identify unusual managerial remuneration

---

## 🧪 Sample Use Cases

* Analyze 5 annual reports to compare **debt-to-equity ratios**
* Identify suspicious **related-party transactions**
* Audit **CFO to Net Profit** conversion trends
* Track increase in **Key Managerial Personnel (KMP)** pay

---

##  License

MIT License – Feel free to use, modify, and share!

---
