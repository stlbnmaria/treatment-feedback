# Treatment Feedback

## Description
The project was about identifying topics from patients' comments with two different diseases (Crohn's Disease and Ulcerative Colitis) with different treatments. The team extracted topics for phrases (incl. classifying the sentiment), assigned markers to topics, and ran an analysis of treatment changes.

## Setup
The project was tested on `3.10 <= Python <= 3.11`.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To install requirements:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

It is assumed that the user has the raw data stored in a folder called `data` which is located in the root.

## Run Pipeline
To run the main pipeline execute the following command in your terminal:
```bash
python main.py
```
Additionally the treatment evolution and the generation of word cloud data can be run seperately.
