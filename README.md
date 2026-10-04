# MIRACL — Metadata Indexing and Retrieval for Automated Context Labelling

> **A context-aware document retrieval and question-answering system that combines document structure, metadata, semantic embeddings, vector search, and Retrieval-Augmented Generation (RAG).**

MIRACL is designed to make information retrieval from complex and unstructured PDF documents more meaningful.

Instead of treating a document as a collection of plain-text fragments, MIRACL preserves its structural context — including headings, sections, subsections, pages, and document hierarchy — and combines this information with semantic vector retrieval.

The retrieved context is then provided to a language model to generate a grounded answer based on the source document.

---

## Why MIRACL?

Modern documents contain more information than just text.

A PDF may contain:

- Titles
- Headings
- Sections
- Subsections
- Paragraphs
- Lists
- Tables
- Page information
- Hierarchical relationships
- Formatting and structural information

A conventional text extraction pipeline can flatten much of this information.

Once the structure is lost, retrieval can return text that is semantically similar but lacks the context needed to understand its meaning.

### MIRACL takes a different approach.

It follows the principle:

> **Preserve document structure first. Retrieve meaning second. Generate answers last.**

This allows the system to combine:

**Structural context + Metadata + Semantic similarity + LLM reasoning**

rather than relying on semantic similarity alone.

---

# Core Idea

The central idea behind MIRACL is to transform a document from:

```text
Raw PDF
   ↓
Plain Text
```

into a structured semantic knowledge layer:

```text
PDF
 ↓
Document Structure
 ↓
Metadata
 ↓
Contextual Chunks
 ↓
Semantic Embeddings
 ↓
Vector Index
 ↓
Context-Aware Retrieval
 ↓
Grounded Answer
```

The document remains the **source of truth**.

The retrieval system finds the relevant information.

The language model converts the retrieved information into a natural-language answer.

---

# Architecture

```text
                         PDF DOCUMENT
                              │
                              ▼
                  Adobe PDF Extract API
                              │
                              ▼
                    Structured JSON
                              │
                              ▼
                    Metadata Extraction
                              │
                              ▼
                    Contextual Chunking
                              │
                              ▼
                   LlamaIndex Documents
                              │
                              ▼
                    BGE Embeddings
                              │
                              ▼
                         ChromaDB
                              │
                              │
                     ┌────────┘
                     │
                User Query
                     │
                     ▼
              Query Embedding
                     │
                     ▼
            Semantic Retrieval
                     │
                     ▼
          Relevant Context + Metadata
                     │
                     ▼
               LangChain RAG
                     │
                     ▼
             Llama 3.2 via Ollama
                     │
                     ▼
               Grounded Answer
```

---

# Key Innovation

MIRACL's main strength is **context-aware retrieval**.

Traditional retrieval can represent content approximately as:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

MIRACL instead preserves relationships such as:

```text
Projects
│
├── Nike Sales Analysis
│   ├── Project description
│   ├── Data preparation
│   └── Dashboard development
│
└── Frozen Food ERP System
    ├── Architecture
    ├── Inventory
    └── Billing
```

This hierarchy is carried into the retrieval process through metadata and contextual chunking.

As a result, retrieved information retains additional meaning about:

- Where it came from
- Which section it belongs to
- Which subsection it belongs to
- What type of document element it represents

This is particularly useful for documents where **context and hierarchy matter as much as the text itself**.

---

# 1. Structural PDF Extraction

MIRACL uses the **Adobe PDF Extract API** to obtain structured information from PDF documents.

The extraction process can identify elements such as:

- Text
- Titles
- Headings
- Paragraphs
- Lists
- Tables
- Figures
- Pages
- Document paths
- Hierarchical structure

The result is represented as structured JSON.

For example, a PDF element can contain information such as:

```json
{
    "Page": 0,
    "Path": "//Document/Sect[5]/Sect/H2",
    "Text": "Nike Sales Analysis"
}
```

The `Path` provides structural information that can be used to understand where an element belongs within the document.

### Why this matters

MIRACL does not treat PDF extraction as merely a text-copying operation.

The structural information becomes part of the retrieval pipeline.

---

# 2. Metadata Engineering

After extracting the document structure, MIRACL creates retrieval-oriented metadata.

A processed element can contain:

```text
text
page
element_type
section
subsection
path
```

For example:

```json
{
    "text": "Nike Sales Analysis",
    "page": 0,
    "element_type": "subheading",
    "section": "Projects",
    "subsection": "Nike Sales Analysis",
    "path": "//Document/Sect[5]/Sect/H2"
}
```

This metadata provides contextual information that a plain text representation would lose.

### Metadata allows MIRACL to answer questions such as:

- Where did this information come from?
- Which section contains it?
- Which subsection does it belong to?
- What type of document element is it?

---

# 3. Contextual Chunking

MIRACL uses the extracted document hierarchy to create contextual chunks.

Instead of blindly splitting text into fixed-size fragments, the system uses structural boundaries such as:

- H1 sections
- H2 subsections
- Related content under those headings

For example:

```text
Projects
   ↓
Nike Sales Analysis
   ↓
Project description
Data preparation
Dashboard development
```

can be treated as a meaningful contextual unit.

This helps prevent important relationships from being separated during chunking.

---

# 4. Semantic Embeddings

MIRACL uses:

**BAAI/bge-small-en-v1.5**

for semantic embedding generation.

The model converts text into a numerical vector representation.

The current implementation produces:

```text
384-dimensional embeddings
```

For document chunks:

```text
Contextual Chunk
      ↓
BGE
      ↓
384-dimensional Vector
```

For user queries:

```text
User Query
      ↓
BGE
      ↓
384-dimensional Vector
```

The same embedding model is used for both documents and queries so that they can be compared within the same semantic vector space.

---

# 5. Semantic Retrieval with ChromaDB

The generated embeddings are stored in **ChromaDB** along with the original chunk text and associated metadata.

Conceptually:

```text
Document Chunk
      │
      ├── Text
      ├── Embedding
      └── Metadata
              │
              ▼
           ChromaDB
```

When a user asks a question:

```text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Most Relevant Chunks
```

This allows MIRACL to retrieve information based on **semantic meaning**, rather than requiring an exact keyword match.

---

# 6. LlamaIndex Retrieval Layer

MIRACL uses **LlamaIndex** as the indexing and retrieval layer connected to the vector store.

The retriever receives the user's query and identifies the most relevant document nodes.

For example:

```python
retriever = llama_index.as_retriever(
    similarity_top_k=3
)

retrieved_nodes = retriever.retrieve(query)
```

The retrieved nodes contain both content and metadata.

This allows the system to preserve the relationship between:

```text
Answer
  ↓
Retrieved Content
  ↓
Document Section
  ↓
Document Structure
```

---

# 7. Retrieval-Augmented Generation

MIRACL uses Retrieval-Augmented Generation (RAG) to connect semantic retrieval with language generation.

The process is:

```text
User Question
      ↓
Query Embedding
      ↓
Semantic Retrieval
      ↓
Top Relevant Chunks
      ↓
Retrieved Context
      ↓
RAG Prompt
      ↓
Language Model
      ↓
Grounded Answer
```

The language model does not need to independently know the contents of the uploaded document.

Instead, MIRACL retrieves the relevant information and provides it as context.

---

# 8. Grounded Answer Generation

The generation layer uses:

- **LangChain** for prompt orchestration
- **Ollama** for local model execution
- **Llama 3.2 3B** as the language model

The prompt instructs the model to answer using the retrieved document context.

This creates a clear separation of responsibilities:

```text
Document
   │
   └── Source of Truth

Retriever
   │
   └── Finds Relevant Information

LLM
   │
   └── Generates Natural-Language Answer
```

The LLM is therefore not treated as the source of document-specific facts.

---

# Grounding and Hallucination Control

One of the important characteristics of MIRACL is its ability to distinguish between information that exists in the document and information that does not.

### Example

If the document contains:

```text
Nike Sales Analysis

Analyzed a 3,000-record Nike retail sales dataset
using Python and Power BI...
```

and the user asks:

> What did I do in the Nike Sales Analysis project?

MIRACL retrieves the relevant project context and generates an answer based on it.

However, if the user asks:

> What is my favorite programming language?

and the document does not contain that information, MIRACL is instructed not to invent an answer.

This provides a more controlled RAG behavior:

```text
Information exists
       ↓
Retrieve
       ↓
Generate answer

Information does not exist
       ↓
Do not invent
       ↓
State that it is not available
```

---

# Why This Architecture Matters

MIRACL demonstrates an important principle in document AI:

> **Good RAG is not only about selecting an embedding model or connecting an LLM to a vector database. The quality of retrieval depends heavily on how the original document is represented before retrieval begins.**

MIRACL therefore focuses on the complete pipeline:

```text
Document Representation
        ↓
Metadata
        ↓
Context Preservation
        ↓
Semantic Representation
        ↓
Retrieval
        ↓
Grounded Generation
```

This makes the system more context-aware than a simple:

```text
PDF → Text → Chunk → Embedding → Vector DB → LLM
```

pipeline.

---

# Technology Stack

| Technology | Role |
|---|---|
| **Python** | Core implementation |
| **Adobe PDF Extract API** | Structural PDF extraction |
| **BAAI/bge-small-en-v1.5** | Semantic embedding model |
| **ChromaDB** | Vector storage and similarity search |
| **LlamaIndex** | Document indexing and retrieval |
| **LangChain** | RAG orchestration and prompt construction |
| **Ollama** | Local LLM runtime |
| **Llama 3.2 3B** | Answer generation |
| **Jupyter Notebook** | Development and validation |

---

# End-to-End Example

Consider the question:

> **What technologies did I use for the Frozen Food ERP project?**

MIRACL processes the query as follows:

```text
"What technologies did I use for
the Frozen Food ERP project?"
                │
                ▼
        BGE Query Embedding
                │
                ▼
        ChromaDB Similarity Search
                │
                ▼
     ERP Project Context Retrieved
                │
                ▼
         LlamaIndex Retriever
                │
                ▼
          LangChain Prompt
                │
                ▼
       Llama 3.2 / Ollama
                │
                ▼
        Grounded Answer
```

The system retrieves the relevant project context instead of searching for an exact phrase alone.

---

# Validation

The implementation has been tested across multiple stages of the pipeline.

### Document Processing

- PDF successfully processed
- Structured JSON successfully generated
- Document elements successfully identified
- Sections and subsections successfully detected
- Contextual chunks successfully created

### Retrieval

- Document chunks embedded using BGE
- 384-dimensional vector generation verified
- Vectors persisted in ChromaDB
- Query embeddings generated using the same embedding model
- Semantic similarity retrieval verified
- LlamaIndex retrieval verified

### Generation

- LangChain prompt construction verified
- Ollama successfully connected
- Llama 3.2 3B successfully generated responses
- Context-grounded answers verified
- Unsupported questions tested successfully

---

# Project Significance

MIRACL explores the intersection of:

- **Document Intelligence**
- **Semantic Search**
- **Metadata Engineering**
- **Vector Databases**
- **Information Retrieval**
- **Retrieval-Augmented Generation**
- **Large Language Models**

The project demonstrates how unstructured documents can be transformed into a structured semantic retrieval system while preserving information about their original organization.

The key idea is simple:

> **Don't throw away the document's structure before trying to understand its meaning.**

---

# Future Directions

Possible extensions include:

- Hybrid keyword + semantic retrieval
- Retrieval reranking
- Multi-document knowledge bases
- Improved hierarchical chunking
- Metadata-aware filtering
- Page-level source citations
- Retrieval evaluation benchmarks
- Document comparison
- Table-aware retrieval
- Multimodal document understanding
- Improved grounding and answer verification

---

# Project Structure

```text
MIRACL/
│
├── MIRACL_Development.ipynb
├── MIRACL_Clean.ipynb
│
├── ...
│
└── README.md
```

The development notebook documents the experimental and validation process, while the cleaned notebook represents the finalized reproducible pipeline.

---

# Key Takeaway

MIRACL is not simply:

```text
PDF + Embeddings + LLM
```

It is a complete context-aware retrieval pipeline:

```text
PDF
 ↓
Structure Extraction
 ↓
Metadata Engineering
 ↓
Contextual Chunking
 ↓
Semantic Embeddings
 ↓
Vector Search
 ↓
Context-Aware Retrieval
 ↓
RAG
 ↓
Grounded Generation
```

By preserving document structure and enriching retrieved content with metadata, MIRACL aims to make semantic retrieval more contextual, interpretable, and useful for real-world document question answering.

---

## Author

**Muhammad Thameem S A**

BTech — Artificial Intelligence & Data Science

GitHub: **Muhammadthameem**

---

## Keywords

`RAG` · `Retrieval-Augmented Generation` · `Semantic Search` · `Document Intelligence` · `Vector Database` · `Embeddings` · `Metadata Extraction` · `Contextual Chunking` · `LlamaIndex` · `LangChain` · `ChromaDB` · `Ollama` · `LLM` · `Python`
