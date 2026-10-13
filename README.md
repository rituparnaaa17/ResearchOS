# ResearchOS – AI Research Assistant

A research paper search and similarity analysis dashboard built with **Streamlit**, **TF-IDF**, and **cosine similarity**.

## Features
- 🔍 Search scientific papers by research topic
- 📄 Upload a PDF and find similar papers automatically
- 📊 View cosine similarity scores for each result
- 🔗 Direct arXiv links for every paper
- 🖥️ ResearchOS-style clean dashboard UI

## Tech Stack
- Python 3.x
- Streamlit
- scikit-learn (TF-IDF + cosine similarity)
- pandas
- PyPDF2

## Setup

```bash
pip install streamlit pandas scikit-learn PyPDF2
```

> **Note:** The dataset files (`*.csv`, `*.npz`, `*.pkl`) are not included in this repository due to size limits.  
> Run `Preprocessing.ipynb` to regenerate the TF-IDF model from your own arXiv dataset.

## Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
ResearchOS/
├── app.py                  # Main Streamlit application
├── Preprocessing.ipynb     # Dataset preprocessing notebook
├── .gitignore
└── README.md
```
