"""
nlp_utils.py
------------
Core classical NLP utilities for the Smart Academic Email Generator.

This module implements every "traditional NLP" stage required by the
project brief:
    1. Text cleaning
    2. Tokenization
    3. Stopword removal
    4. Lemmatization
    5. Keyword extraction (TF-based + POS filtering)
    6. Named Entity Recognition (spaCy, with a regex fallback)
    7. POS tagging
    8. Urgency detection (lexicon + rule based)
    9. Very small rule-based grammar clean-up

No LLM / generative API is used anywhere in this file. Everything here is
either a hand written rule, a lexicon lookup, or a classical statistical
NLP technique (spaCy's NER model is a CRF/transition-based tagger, not a
generative LLM).
"""

import re
import string
from collections import Counter

# --------------------------------------------------------------------------
# Optional heavy dependencies are imported defensively so that the whole
# application keeps working (in degraded mode) even on a machine where the
# spaCy English model or NLTK corpora have not been downloaded yet. This is
# important for a lab-viva laptop that may not have internet access.
# --------------------------------------------------------------------------
try:
    import spacy
    try:
        NLP_SPACY = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded -> run "python -m spacy download en_core_web_sm"
        NLP_SPACY = None
        SPACY_AVAILABLE = False
except ImportError:
    spacy = None
    NLP_SPACY = None
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Attempt silent corpus download the first time the app runs.
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    NLTK_AVAILABLE = True
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    LEMMATIZER = None


# A small built-in stopword list used as a fallback if NLTK corpora are not
# available on the grading machine. Keeping this local guarantees the
# pipeline never crashes during a live viva demo.
FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "and", "or", "but", "if", "then", "so", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "off", "over", "under", "again", "further",
    "this", "that", "these", "those", "not", "no", "nor", "too", "very",
    "can", "will", "just", "should", "now", "do", "does", "did", "have",
    "has", "had", "having",
}


# --------------------------------------------------------------------------
# 1. TEXT CLEANING
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Lowercases the text, strips extra whitespace, removes stray punctuation
    that is not meaningful for keyword extraction, but keeps digits (dates,
    roll numbers) and single hyphens/apostrophes intact.

    Why: Raw user input is often informal ("hospital!!, urgent??") and must
    be normalised before tokenization so that downstream statistical steps
    (frequency counts, matching) are not thrown off by noise characters.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\r\n]+", " ", text)
    # keep letters, digits, spaces, apostrophes and hyphens
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# 2. TOKENIZATION
# --------------------------------------------------------------------------
def tokenize(text: str):
    """
    Splits cleaned text into word tokens.
    Uses NLTK's word_tokenize when available (handles contractions and
    punctuation edge cases better); falls back to a simple regex splitter.
    """
    if not text:
        return []
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return text.split()


# --------------------------------------------------------------------------
# 3. STOPWORD REMOVAL
# --------------------------------------------------------------------------
def remove_stopwords(tokens):
    """
    Filters out high-frequency, low-information words (stopwords) so that
    keyword extraction and intent classification focus on content-bearing
    words only.
    """
    if NLTK_AVAILABLE:
        try:
            stop_set = set(nltk_stopwords.words("english"))
        except Exception:
            stop_set = FALLBACK_STOPWORDS
    else:
        stop_set = FALLBACK_STOPWORDS
    return [t for t in tokens if t not in stop_set and t not in string.punctuation and len(t) > 1]


# --------------------------------------------------------------------------
# 4. LEMMATIZATION
# --------------------------------------------------------------------------
def lemmatize(tokens):
    """
    Reduces each token to its dictionary (base) form, e.g. "missed" -> "miss",
    "activities" -> "activity". This normalises different inflections of the
    same word so keyword matching against our intent lexicons is robust.
    """
    if NLTK_AVAILABLE and LEMMATIZER is not None:
        try:
            return [LEMMATIZER.lemmatize(t, pos="v") if len(t) > 2 else t for t in tokens]
        except Exception:
            pass
    if SPACY_AVAILABLE:
        doc = NLP_SPACY(" ".join(tokens))
        return [tok.lemma_ for tok in doc]
    return tokens  # graceful no-op fallback


# --------------------------------------------------------------------------
# 5. POS TAGGING
# --------------------------------------------------------------------------
def pos_tag_tokens(text: str):
    """
    Returns a list of (token, POS tag) tuples. Used to bias keyword
    extraction towards nouns/proper-nouns/adjectives, which carry the most
    "reason" information (e.g. "hospital"=NOUN, "urgent"=ADJ).
    """
    if SPACY_AVAILABLE:
        doc = NLP_SPACY(text)
        return [(tok.text, tok.pos_) for tok in doc]
    if NLTK_AVAILABLE:
        try:
            return nltk.pos_tag(word_tokenize(text))
        except Exception:
            pass
    # Fallback: everything treated as NOUN so keyword extraction still works
    return [(t, "NOUN") for t in text.split()]


# --------------------------------------------------------------------------
# 6. KEYWORD EXTRACTION
# --------------------------------------------------------------------------
def extract_keywords(raw_text: str, top_n: int = 12):
    """
    Classical (non-neural) keyword extraction pipeline:
        clean -> tokenize -> stopword removal -> lemmatize -> POS filter
        -> term-frequency ranking.

    We keep tokens whose POS tag is NOUN, PROPN, ADJ, or VERB (content
    words) and rank them by frequency. Multi-word "keyword phrases" typed
    by the user on separate lines are preserved as phrases too, since a
    phrase like "cousin accident" is more informative than the two words
    split apart.
    """
    if not raw_text:
        return []

    # Preserve line-separated phrases as extra candidate keyphrases
    line_phrases = [clean_text(line) for line in raw_text.split("\n") if line.strip()]

    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    tokens_ns = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_ns)

    pos_tags = dict(pos_tag_tokens(cleaned))
    content_pos = {"NOUN", "PROPN", "ADJ", "VERB", "VB", "VBD", "VBN", "NN", "NNS", "JJ", "NNP"}

    filtered = [w for w in lemmas if pos_tags.get(w, "NOUN") in content_pos or pos_tags.get(w, "") == ""]
    if not filtered:
        filtered = lemmas  # if POS filter removed everything, keep lemmas

    freq = Counter(filtered)
    ranked_words = [w for w, _ in freq.most_common(top_n)]

    # Merge in short informative multi-word phrases (2-3 tokens) not already covered
    keyphrases = []
    for phrase in line_phrases:
        if phrase and phrase not in ranked_words and len(phrase.split()) <= 4:
            keyphrases.append(phrase)

    combined = ranked_words + [p for p in keyphrases if p not in ranked_words]
    return combined[:top_n]


# --------------------------------------------------------------------------
# 7. NAMED ENTITY RECOGNITION
# --------------------------------------------------------------------------
DATE_REGEX = re.compile(
    r"\b(\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}|"
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",
    re.IGNORECASE,
)
DURATION_REGEX = re.compile(r"\b(\d+)\s*(day|days|week|weeks|hour|hours)\b", re.IGNORECASE)


def extract_entities(raw_text: str):
    """
    Extracts named entities relevant to an academic-leave email: DATE,
    PERSON, ORG, GPE, and duration expressions ("2 days").

    Uses spaCy's statistical NER model when available (a classical
    sequence-labelling model, not an LLM). If the model is not installed on
    the machine, falls back to hand-written regex rules for dates and
    durations, which is enough to keep the pipeline explainable and
    functional for the viva.
    """
    entities = {"DATE": [], "PERSON": [], "ORG": [], "GPE": [], "DURATION": []}

    if SPACY_AVAILABLE:
        doc = NLP_SPACY(raw_text)
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

    # Regex fallback / supplement for dates and durations (spaCy sometimes
    # misses informal date formats like "21 September")
    for m in DATE_REGEX.finditer(raw_text):
        val = m.group(0)
        if val not in entities["DATE"]:
            entities["DATE"].append(val)
    for m in DURATION_REGEX.finditer(raw_text):
        val = m.group(0)
        if val not in entities["DURATION"]:
            entities["DURATION"].append(val)

    return entities


# --------------------------------------------------------------------------
# 8. URGENCY DETECTION
# --------------------------------------------------------------------------
URGENT_LEXICON = {
    "urgent", "emergency", "immediately", "asap", "critical", "serious",
    "hospitalized", "hospitalised", "admitted", "accident", "severe",
    "immediate", "today", "right now",
}


def detect_urgency(raw_text: str) -> str:
    """
    Rule-based urgency classifier. Scans the (lower-cased) raw text for
    words from a hand-curated urgency lexicon and returns "High" or
    "Normal". A frequency-weighted score could be swapped in later, but a
    lexicon match is transparent and easy to explain in a viva.
    """
    text_l = raw_text.lower()
    hits = sum(1 for w in URGENT_LEXICON if w in text_l)
    return "High" if hits > 0 else "Normal"


# --------------------------------------------------------------------------
# 9. SMALL RULE-BASED GRAMMAR / TEXT POLISH
# --------------------------------------------------------------------------
def capitalize_sentences(text: str) -> str:
    """
    Rule-based post-processing: capitalises the first letter of every
    sentence and the pronoun "I", and ensures sentences end with proper
    punctuation. This is a lightweight substitute for a full grammar
    checker and keeps the generated email looking professional.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    fixed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        s = re.sub(r"\bi\b", "I", s)
        if not s.endswith((".", "!", "?")):
            s += "."
        fixed.append(s)
    return " ".join(fixed)


def title_case_name(name: str) -> str:
    """Normalises a proper noun (teacher/student name) to Title Case."""
    return " ".join(w.capitalize() for w in name.strip().split())
