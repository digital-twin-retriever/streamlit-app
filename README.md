# Digital Twin Retriever

AI-powered chatbot for searching and summarizing clinical case reports.

## Introduction

**Digital Twin Retriever** is an AI-powered chatbot designed to help users search, retrieve, and summarize real-world clinical cases.  
By combining a rich case database, semantic retrieval techniques, and generative AI, it supports healthcare professionals in making evidence-informed decisions.

The tool leverages the [MultiCaRe dataset](https://zenodo.org/records/14994046), an open-access collection of over 90,000 de-identified clinical case reports sourced from PubMed Central.

## Features
- 🧠 Semantic search of real-world clinical cases based on user queries.
- 📚 Summarized answers grounded in retrieved case reports.
- 🔎 Ability to export similar cases into downloadable CSV files.
- 🤖 Powered by advanced retrieval techniques and generative AI models.
- 🌐 Clean Streamlit web interface for easy interaction.


## How It Works
- Enter a clinical case description or a specific clinical question.
- The AI will search for the most relevant cases using semantic embeddings.
- Retrieved cases are analyzed to generate a structured, evidence-based response.
- You can export similar cases for further review or analysis.


## Live Demo
🚀 Try **Digital Twin Retriever** live:
👉 [Access the App on Streamlit Cloud](https://digital-twin-retriever.streamlit.app/)


## Screenshots
![Main Page](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtw-main-page.webp)

![Case Description Example](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtr-case-description.webp)

![Formatted References](https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/main/img/dtr-text-references.webp)


## Promotional Video

📺 **Watch the promotional video**:  
[Digital Twin Retriever - YouTube](https://www.youtube.com/watch?v=zO4E0DfTQuY)


## Quick Start

1. Clone the repository:

    ```bash
    git clone https://github.com/digital-twin-retriever/streamlit-app.git
    cd streamlit-app
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set your Google API Key (`.streamlit/secrets.toml`):

    ```toml
    [GOOGLE_API_KEY]
    GOOGLE_API_KEY = "your-google-api-key-here"
    ```

4. Run the app:

    ```bash
    streamlit run app.py
    ```

## Authors

- [María Carolina González Galtier, MD, MA](https://www.linkedin.com/in/carogaltier/)
- [Mauro Andrés Nievas Offidani, MD, MSc](https://www.linkedin.com/in/mauronievasoffidani/)


## License

This project is released under the [CC0-1.0 license](LICENSE).
