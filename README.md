# Smart Academic Email Generator using NLP

Converts short keywords (e.g. *"hospital, mother admitted, urgent, missed
practical, sorry"*) into a complete, professional academic email — using
**classical NLP techniques only**. No OpenAI / Claude / Gemini API or any
Large Language Model is used to generate the email content.

---

## 1. Folder Structure

```
SmartAcademicEmailGenerator/
│
├── app.py                 # Streamlit front-end
├── email_generator.py     # Orchestrates the full NLP pipeline
├── templates.py           # Rule-based NLG templates (15 intents)
├── intent_classifier.py   # Lexicon + TF-IDF intent classification
├── nlp_utils.py            # Cleaning, tokenization, POS, NER, keyword extraction
├── requirements.txt
├── README.md               # (this file)
├── assets/                 # Screenshots / static assets
└── notebook.ipynb          # Full documented notebook version of the project
```

---

## 2. NLP Pipeline

```
User Input
   │
   ▼
Text Cleaning            (regex normalisation)
   │
   ▼
Tokenization              (NLTK word_tokenize)
   │
   ▼
Stopword Removal          (NLTK stopword corpus)
   │
   ▼
Lemmatization              (WordNet Lemmatizer / spaCy)
   │
   ▼
Keyword Extraction         (POS-filtered term frequency)
   │
   ▼
Intent Classification      (keyword lexicon vote + TF-IDF cosine similarity)
   │
   ▼
Entity Extraction          (spaCy NER + regex fallback: DATE, PERSON, ORG, GPE, DURATION)
   │
   ▼
Reason Classification      (rule-based reason-phrase construction)
   │
   ▼
Template Selection         (dictionary lookup by predicted intent)
   │
   ▼
Dynamic Email Generation   (slot-filling NLG)
   │
   ▼
Subject Generation
   │
   ▼
Output (Subject + Body)
```

Every stage is implemented as a hand-written rule, a lexicon lookup, or a
classical statistical NLP/ML technique (spaCy's NER model, NLTK's
tokenizer/lemmatizer, scikit-learn's `TfidfVectorizer` + cosine similarity).
**No generative language model is used anywhere in the pipeline.**

---

## 3. Supported Email Intents (15)

1. Medical Leave
2. Family Emergency
3. Club Activity
4. Technical Competition
5. Hackathon
6. Sports Event
7. NSS Activity
8. Industrial Visit
9. Internship
10. Permission Request
11. Assignment Extension
12. Project Deadline Extension
13. Apology for Absence
14. Thank You Email
15. Meeting Request

---

## 4. Installation Guide

### 4.1 Prerequisites
- Python 3.11 (3.10+ also works)
- pip

### 4.2 Steps

```bash
# 1. Clone / copy the project folder, then move into it
cd SmartAcademicEmailGenerator

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model (one-time, needs internet)
python -m spacy download en_core_web_sm

# 5. (First run only) NLTK will auto-download 'punkt', 'stopwords',
#    'wordnet' the first time nlp_utils.py runs. If your machine has no
#    internet access at that point, run this once manually beforehand:
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> **Note:** The application is designed to **degrade gracefully**. If the
> spaCy model or NLTK corpora are not available on the machine (e.g. during
> an offline viva), `nlp_utils.py` automatically falls back to lightweight
> regex-based tokenisation/NER so the app keeps working — with slightly
> reduced NER accuracy.

### 4.3 Run the notebook (development / documentation version)

```bash
jupyter notebook notebook.ipynb
```

### 4.4 Run the Streamlit web app

```bash
streamlit run app.py
```

Open the printed local URL (typically `http://localhost:8501`) in your
browser.

---

## 5. Hosting Guide — Streamlit Community Cloud

1. Push the project folder to a **public (or private) GitHub repository**,
   including `app.py`, `email_generator.py`, `templates.py`,
   `intent_classifier.py`, `nlp_utils.py`, and `requirements.txt`.
2. Add a `packages.txt` file (optional but recommended) if you want the
   deployment to pre-fetch the spaCy model; alternatively add this line at
   the top of `app.py`'s import block, or run it once via a `setup.sh` /
   `postBuild` step:
   ```python
   import spacy
   try:
       spacy.load("en_core_web_sm")
   except OSError:
       from spacy.cli import download
       download("en_core_web_sm")
   ```
3. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.
4. Click **"New app"**, select your repository, branch, and set the main
   file path to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` automatically.
6. Once deployed, you'll get a public URL like
   `https://your-app-name.streamlit.app` that you can share/submit.

---

## 6. Future Scope

- Rule-based/statistical grammar checker integration for more robust
  polishing.
- Train a small supervised classifier (Naive Bayes / SVM on TF-IDF
  features) on a larger real dataset of student emails, replacing the
  hand-curated lexicon while still avoiding LLMs.
- Multilingual keyword support (Hindi/Marathi) via classical multilingual
  NLP tools.
- Direct SMTP email-sending integration from the app.
- Per-student history/log of generated emails.

---

## 7. Viva Questions & Answers

**Q1. Why does this project avoid using ChatGPT/OpenAI/LLM APIs?**
A: The objective is to demonstrate mastery of *classical* NLP techniques —
tokenization, stopword removal, lemmatization, POS tagging, NER, and
template-based NLG — rather than depending on a black-box generative model.
Every output can be traced back to an explainable rule or a classical
statistical computation.

**Q2. What NLP techniques are used in this project?**
A: Text cleaning (regex normalisation), tokenization (NLTK), stopword
removal (NLTK corpus), lemmatization (WordNet Lemmatizer / spaCy), POS
tagging (spaCy/NLTK), keyword extraction (POS-filtered term-frequency
ranking), Named Entity Recognition (spaCy's pretrained statistical NER
model, with a regex fallback), and template-based Natural Language
Generation.

**Q3. How is the intent of the email determined?**
A: Two complementary signals are combined: (1) a hand-curated
**keyword-lexicon vote**, where extracted keywords are matched against
each intent's lexicon, and (2) a **TF-IDF + cosine similarity** score
(scikit-learn) comparing the raw input against representative example
sentences for each intent. The two scores are combined with fixed weights
(60% lexicon, 40% TF-IDF) to select the final intent and produce a
confidence score.

**Q4. Is spaCy's NER model an LLM?**
A: No. spaCy's `en_core_web_sm` NER component is a small, transition-based
statistical sequence-labelling model (roughly comparable in spirit to a
CRF), trained specifically for the narrow task of identifying entity spans.
It is not a generative language model and cannot compose free text.

**Q5. How does the system generate grammatically correct sentences without
an LLM?**
A: Through **template-based Natural Language Generation (NLG)** — each
intent maps to a hand-written, grammatically correct template with
placeholders. The pipeline only *fills in the blanks* (teacher name, reason
phrase, date, urgency phrasing) using rule-based slot-filling; it never
freely generates novel sentence structure.

**Q6. What happens if the keyword input doesn't clearly match any intent?**
A: The system falls back to the general-purpose "Apology for Absence"
template with a moderate default confidence score, ensuring the app never
crashes or produces an empty output.

**Q7. How is urgency detected?**
A: A hand-curated lexicon of urgency-indicating words (e.g. "urgent",
"emergency", "immediately", "hospitalized", "accident") is matched against
the lower-cased input text. A match sets urgency to "High" and inserts an
extra urgency sentence into the email body.

**Q8. How are dates and durations extracted?**
A: Primarily via spaCy's NER (`DATE` entity label). Because informal date
formats (e.g. "21 September") are sometimes missed by the statistical
model, a regex-based fallback/supplement specifically checks for common
date and duration patterns ("21 September", "12/09/2025", "2 days").

**Q9. What is TF-IDF and why is cosine similarity used here?**
A: TF-IDF (Term Frequency–Inverse Document Frequency) represents text as a
vector where each dimension's weight reflects how important a term is to a
document relative to a corpus. Cosine similarity measures the angle between
two such vectors (1 = identical direction/very similar, 0 = unrelated).
Here, it measures how similar the user's raw input is to each intent's
reference example sentences, giving a continuous, interpretable confidence
score.

**Q10. How would you extend this project to support more intents?**
A: Add a new entry to `INTENT_LEXICON` (keywords + example sentences) in
`intent_classifier.py`, add a subject/body template pair in `templates.py`,
and optionally extend the reason-phrase rules in
`email_generator.py::_build_reason_phrase`. No retraining is required
because the system is rule-based/lexicon-based rather than a trained
classifier.

**Q11. Why combine a rule-based approach with TF-IDF instead of using only
one?**
A: Short keyword-style input (as specified in the input format) favours
exact/near-exact lexicon matches, which the rule-based vote captures well.
TF-IDF cosine similarity adds robustness when the input is phrased as full
sentences rather than isolated keywords, and it provides a genuinely
continuous confidence score that a simple vote count cannot.

**Q12. What are the limitations of this system?**
A: Grammar correction is limited to simple rule-based capitalisation and
punctuation fixes; it cannot fix deep grammatical errors. Intent accuracy
depends on lexicon/example coverage. Very novel or ambiguous situations
outside the 15 supported intents may not be classified precisely.

---

## 8. License / Academic Use

This project was built as an academic NLP Lab mini-project. Feel free to
extend it for coursework and viva demonstrations.
