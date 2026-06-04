# Digital Twin Retriever

Retrieval-augmented AI system for semantic search and exploration of clinical case reports.

## Introduction

**Digital Twin Retriever** is an AI-powered clinical retrieval system designed to help users search, retrieve, and summarize real-world clinical case reports.

By combining semantic retrieval methods with generative AI, the application supports exploration of clinical evidence grounded in published case reports.

The system leverages the [MultiCaRe dataset](https://zenodo.org/records/14994046), an open-access collection of more than 90,000 de-identified clinical case reports sourced from PubMed Central.

---

## Technical Overview

Digital Twin Retriever implements a lightweight retrieval-augmented generation (RAG) workflow focused on clinical case exploration.

The system combines:

* Semantic similarity search using dense vector embeddings
* Context-aware retrieval from clinical case report data
* Conversational memory handling for follow-up questions
* Generative answer synthesis grounded in retrieved evidence
* Automatic PMCID extraction and citation formatting
* Exportable retrieval results for downstream analysis

The application was designed as a practical demonstration of how retrieval systems and generative AI can support exploration of real-world clinical evidence in healthcare contexts.

---

## Features

* Semantic search of clinical case reports using natural language queries
* Retrieval-grounded AI-generated responses
* Automatic reference extraction and formatted citations
* Conversational context support for follow-up clinical questions
* CSV export of retrieved similar cases
* Streamlit-based interactive web interface

---

## Tech Stack

* Python
* Streamlit
* Google Gemini API
* Semantic embeddings
* Retrieval-Augmented Generation (RAG)
* Pandas / NumPy
* PubMed Central / MultiCaRe dataset

---

## How It Works

1. The user submits a clinical question or case description.
2. The system generates semantic embeddings for retrieval.
3. Relevant clinical case reports are identified using similarity search.
4. Retrieved evidence is used to generate a grounded response.
5. References are automatically formatted and linked to PubMed Central sources.

---

## Live Demo

Try the application on Streamlit Cloud:

https://digital-twin-retriever.streamlit.app/

---

## Screenshots

### Main Interface

![Main Page](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtw-main-page.webp)

### Clinical Query Example

![Case Description Example](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtr-case-description.webp)

### Automatic Citation Formatting

![Formatted References](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtr-text-references.webp)

---

## Promotional Video

Watch the project overview:

https://www.youtube.com/watch?v=zO4E0DfTQuY

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/digital-twin-retriever/streamlit-app.git
cd streamlit-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "your-google-api-key"
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## Authors

* [María Carolina González Galtier, MD, MA](https://www.linkedin.com/in/carogaltier/)
* [Mauro Andrés Nievas Offidani, MD, MSc](https://www.linkedin.com/in/mauronievasoffidani/)

---

## Disclaimer

This application is intended for research, educational, and demonstration purposes only.

Generated responses are based on retrieved clinical case reports and should not be interpreted as medical advice, diagnostic guidance, or treatment recommendations for real patients.

Clinical decisions should always rely on qualified healthcare professionals and validated clinical guidelines.

---

## License

This project is released under the [CC0-1.0 license](LICENSE).
