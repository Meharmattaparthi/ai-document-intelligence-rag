# 📄 AI Document Intelligence & RAG Assistant

An AI-powered document question-answering system built using **Retrieval-Augmented Generation (RAG)**.

The system processes a document, creates semantic embeddings, retrieves relevant information, reranks the retrieved results, and uses a Gemini LLM to generate grounded answers with document page references.

This project demonstrates an end-to-end AI/ML pipeline for working with structured and unstructured document data.

---

## 🚀 Project Overview

The application allows users to ask natural-language questions about a loaded document.

For demonstration, the project uses a **Metformin drug label from DailyMed/NLM**.

The system follows this pipeline:

```
PDF Document
↓

Text Extraction
↓
Text Cleaning & Chunking
↓
Sentence Transformer Embeddings
↓
Semantic Similarity Retrieval
↓
CrossEncoder Reranking
↓
Top Relevant Document Chunks
↓
Grounded Prompt Construction
↓
Gemini LLM
↓
Answer + Source Pages
```

---

## 🧠 Key Features

* PDF document text extraction
* Text cleaning and section-aware chunking
* Semantic vector embeddings
* Semantic similarity search
* CrossEncoder-based reranking
* Retrieval-Augmented Generation (RAG)
* Gemini LLM integration
* Grounded prompting to reduce unsupported answers
* Document page/source references
* Interactive Streamlit interface
* Retrieval evaluation
* Local retrieval testing without requiring an LLM API call
* Graceful handling of Gemini API quota errors
* **Dynamic PDF Upload:** Upload a new PDF through the Streamlit interface and automatically run the document through text extraction, chunking, embedding generation, semantic retrieval, and CrossEncoder reranking.
* **Document Reset:** Reset the active document back to the bundled Metformin demonstration document.

---

## 🛠️ Technologies Used

### Programming

* Python 3.10+

### Document Processing

* PyMuPDF

### Machine Learning / NLP

* Sentence Transformers --> all-MiniLM-L6-v2
* CrossEncoder --> cross-encoder/ms-marco-MiniLM-L-6-v2
* NumPy

### Generative AI

* Google Gemini API --> gemini-2.5-flash-lite

### Application

* Streamlit

### Configuration

* python-dotenv

---

## 📁 Project Structure

```
 ai-document-intelligence-rag/
 │
 ├── app/
 │   └── app.py
 │
 ├── data/
 │   ├── raw/
 │   └── processed/
 │
 ├── notebooks/
 │
 ├── src/
 │   ├── ingestion/
 │   │   └── document_loader.py
 │   │
 │   ├── processing/
 │   │   └── text_chunker.py
 │   │
 │   ├── embeddings/
 │   │   └── embed_chunks.py
 │   │
 │   ├── retrieval/
 │   │   └── semantic_search.py
 │   │
 │   ├── rag/
 │   │   ├── rag_pipeline.py
 │   │   └── rag_service.py
 │   │
 │   └── evaluation/
 │       ├── evaluate_retrieval.py
 │       ├── evaluate_rag.py
 │       └── rag_evaluation_questions.json
 │
 ├── tests/
 │   └── test_rag_local.py
 │
 ├── .gitignore
 ├── requirements.txt
 └── README.md
```

## ⚙️ Installation

Clone the repository:

```
git clone <your-github-repository-url>
cd ai-document-intelligence-rag
```

Create a virtual environment:

```
python -m venv .venv
```

Activate the environment on Windows:

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🔑 Gemini API Configuration

Create a **.env** file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

The API key is loaded using **python-dotenv**.

**Never commit the '.env' file to GitHub.**

The **.gitignore** configuration excludes **.env** files from version control.

---

## 📄 Document Processing

The current demonstration uses:

```
data/raw/metformin.pdf
```

The document is processed through the following stages.

1. **Text Extraction**

PyMuPDF extracts text from each PDF page while preserving page numbers.

2. **Text Cleaning**

The extracted text is normalized by removing unnecessary spacing and blank lines.

3. **Chunking**

The document is divided into smaller chunks for retrieval.

The chunking process also tracks important document sections such as:

* CONTRAINDICATIONS
* WARNINGS AND PRECAUTIONS
* DOSAGE AND ADMINISTRATION
* DRUG INTERACTIONS
* ADVERSE REACTIONS
* USE IN SPECIFIC POPULATIONS
* CLINICAL PHARMACOLOGY
* CLINICAL STUDIES

---

## 🔎 Retrieval Pipeline

The retrieval system uses a two-stage approach.

### Stage 1 — Semantic Retrieval

Each document chunk is converted into an embedding using:

```
all-MiniLM-L6-v2
```

The query is also embedded and compared with the document embeddings using vector similarity.

The most relevant candidates are selected.

### Stage 2 — CrossEncoder Reranking

The candidate chunks are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This provides a more detailed relevance comparison between:

```
User Query ↔ Retrieved Document Chunk
```

The highest-ranked chunks are then passed to the generation stage.

---

## 🤖 RAG Generation

The retrieved document chunks are inserted into a grounded prompt.

The prompt instructs the LLM to:

* Use only the supplied document context
* Avoid inventing information
* Use relevant retrieved information
* Include important document facts
* State when information is unavailable
* Reference relevant document page numbers

The application then sends the grounded prompt to Gemini.

---

## 💬 Streamlit Application

Run the application with:

```
streamlit run app/app.py
```

The application provides:

* Conversational question input
* Generated answers
* Retrieved source chunks
* Page numbers
* Semantic similarity scores
* Reranker scores
* RAG pipeline details
* Grounded prompt inspection

---

## 🧪 Local Retrieval Testing

The project includes a local test that does not call the Gemini API.

Run:

```
python tests/test_rag_local.py
```

This verifies:

* Embedding model loading
* Reranker loading
* Document data loading
* Semantic retrieval
* CrossEncoder reranking
* Section metadata
* Prompt construction

This allows the retrieval pipeline to be tested even when the Gemini API is unavailable.

---

## 📊 Retrieval Evaluation

A small evaluation set containing **6 questions** was used to evaluate retrieval performance.

The evaluation measures:

### Hit Rate@5

Percentage of questions where at least one expected relevant chunk appeared in the top 5 retrieved results.

> Hit Rate@5: 100.00%

### Average Recall@5

Average percentage of expected relevant chunks retrieved within the top 5 results.

> Average Recall@5: 72.22%

### Mean Reciprocal Rank

Measures how highly the first relevant result appears in the retrieved ranking.

> MRR: 0.78

These results are based on the current six-question evaluation set and should be interpreted as a small project-level evaluation rather than a general benchmark.

---

## 📈 RAG Evaluation

The project also evaluates generated answers against reference answers using a simple keyword-coverage measurement.

Current average reference-answer keyword coverage:

> ~86.67%

This is a simple lexical evaluation metric and should not be interpreted as overall answer accuracy.

The evaluation set currently contains only six questions.

---

## ❓ Example Questions

Example questions that can be asked in the application:

```
What are the contraindications of metformin?

What are the risk factors for lactic acidosis?

What is the recommended starting dose of metformin?

What are the important drug interactions?

What are the recommendations for renal impairment?

What are the common adverse reactions?
```

---

## 🛡️ Error Handling

The application handles common Gemini API failures.

For example, if the Gemini API quota is exhausted, the application displays:

> Answer generation was not completed:
> Gemini API quota is exhausted

The retrieval system can still be tested independently using the local retrieval test.

This separates the local RAG retrieval components from external LLM availability.

---

## ⚠️ Limitations

* The current demonstration uses a single document.
* The retrieval evaluation contains only six questions.
* RAG evaluation metrics are intended for project-level testing rather than production benchmarking.
* Gemini API availability depends on API quota and service availability.
* The current chunking strategy is designed for the demonstration document.
* The application is not intended to provide medical advice.

---

## 🔮 Future Improvements

Potential future improvements include:

* Multi-document ingestion
* Automatic document type detection
* More advanced chunking strategies
* Vector database integration
* Hybrid keyword + semantic retrieval
* Metadata filtering
* Query rewriting
* Improved evaluation datasets
* Automated answer evaluation
* Document upload directly through the UI
* Authentication and user management
* Production deployment
* Monitoring and logging

---

## 🎯 Project Objective

This project demonstrates practical implementation of:

* Machine Learning
* Natural Language Processing
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* Semantic Search
* Vector Embeddings
* CrossEncoder Reranking
* Unstructured Data Processing
* AI Application Development
* Model Evaluation

It is designed as an internship portfolio project demonstrating an end-to-end AI/ML application rather than only a standalone model.

---

## 📌 Disclaimer

The document used in this demonstration contains medical information.

The application is a technical demonstration of document retrieval and question answering and is  **not a medical advice system** .

Always consult an appropriate qualified professional for medical decisions.
