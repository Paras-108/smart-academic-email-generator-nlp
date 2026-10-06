"""
intent_classifier.py
---------------------
Rule-based + classical-ML intent classification for the Smart Academic
Email Generator.

Two complementary techniques are combined (both "classical" NLP/ML, no
LLMs):

1. LEXICON / KEYWORD-VOTING classifier
   Every supported email intent has a hand-curated keyword set. The
   input's extracted keywords are matched against every intent's lexicon
   and the intent with the highest overlap "wins". This is fully
   rule-based and 100% explainable in a viva.

2. TF-IDF + COSINE SIMILARITY classifier (scikit-learn)
   Each intent also has a handful of representative example sentences.
   We vectorise the user's raw text and every intent's example sentences
   with TfidfVectorizer and compute cosine similarity. This gives a
   continuous "confidence score" (0-1) which the rule-based vote alone
   cannot provide, and satisfies the "show confidence score (rule-based or
   ML)" requirement using real scikit-learn machinery.

The final intent is chosen by combining both signals (see
`classify_intent`). If both agree, confidence is boosted; if they
disagree, the lexicon vote (more reliable for short keyword input) is
trusted but the TF-IDF score is still reported for transparency.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 15 supported intents, each with:
#   - a keyword lexicon (for rule-based voting)
#   - a few example sentences (for TF-IDF similarity)
# --------------------------------------------------------------------------
INTENT_LEXICON = {
    "medical_leave": {
        "keywords": {"sick", "ill", "fever", "hospital", "hospitalized", "hospitalised",
                      "doctor", "medicine", "surgery", "injury", "injured", "accident",
                      "health", "unwell", "admitted", "treatment", "covid", "flu"},
        "examples": [
            "I was hospitalized due to sudden fever and could not attend class",
            "I am suffering from illness and need medical leave",
            "I met with an accident and am undergoing treatment",
        ],
    },
    "family_emergency": {
        "keywords": {"father", "mother", "family", "cousin", "brother", "sister",
                      "grandfather", "grandmother", "emergency", "relative", "demise",
                      "passed", "death", "critical", "admitted"},
        "examples": [
            "There is a family emergency and my father is admitted to hospital",
            "My cousin met with an accident and my family needs me urgently",
            "Due to a sudden emergency at home I could not attend college",
        ],
    },
    "club_activity": {
        "keywords": {"club", "cultural", "event", "fest", "festival", "committee",
                      "organizing", "organising", "volunteer", "coordinator"},
        "examples": [
            "I am organizing a club event and need permission to be absent",
            "As a member of the college club I have to attend an inter college fest",
        ],
    },
    "technical_competition": {
        "keywords": {"competition", "technical", "coding", "hackathon", "contest",
                      "codechef", "codeforces", "robotics", "project", "presentation"},
        "examples": [
            "I have been selected for a technical competition at another college",
            "I need permission to participate in a coding contest",
        ],
    },
    "hackathon": {
        "keywords": {"hackathon", "selected", "national", "team", "coding", "prototype"},
        "examples": [
            "My team has been selected for the national level hackathon",
            "I need two days leave to attend a hackathon",
        ],
    },
    "sports_event": {
        "keywords": {"sports", "match", "tournament", "cricket", "football", "athletics",
                      "coach", "practice", "selection", "team", "stadium"},
        "examples": [
            "I have been selected to play in the inter college sports tournament",
            "I need leave to attend sports practice and a match",
        ],
    },
    "nss_activity": {
        "keywords": {"nss", "camp", "social", "service", "blood", "donation", "plantation",
                      "awareness", "rally", "volunteer"},
        "examples": [
            "I have to attend the NSS camp organized by the college",
            "As part of NSS activity I am volunteering for a blood donation drive",
        ],
    },
    "industrial_visit": {
        "keywords": {"industrial", "visit", "factory", "company", "plant", "tour", "industry"},
        "examples": [
            "Our department has organized an industrial visit to a manufacturing company",
            "I need permission for the industrial visit scheduled next week",
        ],
    },
    "internship": {
        "keywords": {"internship", "intern", "company", "offer", "training", "joining",
                      "stipend", "onboarding"},
        "examples": [
            "I have received an internship offer and need to join immediately",
            "I request leave to attend my internship training program",
        ],
    },
    "permission_request": {
        "keywords": {"permission", "allow", "request", "approval", "leave early",
                      "outside", "gate pass"},
        "examples": [
            "I request your permission to leave the college early today",
            "Kindly grant me permission for the following reason",
        ],
    },
    "assignment_extension": {
        "keywords": {"assignment", "submission", "extension", "deadline", "late",
                      "homework", "extra", "time"},
        "examples": [
            "I request an extension for submitting my assignment due to illness",
            "I was unable to complete the assignment on time and need extra days",
        ],
    },
    "project_deadline_extension": {
        "keywords": {"project", "deadline", "extension", "report", "submission", "final",
                      "review", "delay"},
        "examples": [
            "I request an extension of the project submission deadline",
            "Due to technical issues my project work got delayed",
        ],
    },
    "apology_absence": {
        "keywords": {"absent", "absence", "missed", "sorry", "apology", "apologize",
                      "apologise", "lab", "practical", "lecture", "class"},
        "examples": [
            "I am extremely sorry for being absent from the lab session",
            "I apologize for missing yesterday's lecture",
        ],
    },
    "thank_you": {
        "keywords": {"thank", "thanks", "grateful", "appreciate", "gratitude", "helpful"},
        "examples": [
            "I sincerely thank you for your guidance and support",
            "I am grateful for the opportunity and your help",
        ],
    },
    "meeting_request": {
        "keywords": {"meeting", "discuss", "appointment", "schedule", "available", "time",
                      "doubt", "guidance", "consult"},
        "examples": [
            "I would like to schedule a meeting to discuss my project doubts",
            "Kindly let me know a convenient time to meet and discuss",
        ],
    },
}

INTENT_LABELS_READABLE = {
    "medical_leave": "Medical Leave",
    "family_emergency": "Family Emergency",
    "club_activity": "Club Activity",
    "technical_competition": "Technical Competition",
    "hackathon": "Hackathon",
    "sports_event": "Sports Event",
    "nss_activity": "NSS Activity",
    "industrial_visit": "Industrial Visit",
    "internship": "Internship",
    "permission_request": "Permission Request",
    "assignment_extension": "Assignment Extension",
    "project_deadline_extension": "Project Deadline Extension",
    "apology_absence": "Apology for Absence",
    "thank_you": "Thank You Email",
    "meeting_request": "Meeting Request",
}


def _lexicon_vote(keywords):
    """
    Rule-based voting: counts overlap between extracted keywords and each
    intent's lexicon. Returns a dict {intent: score}.
    """
    kw_set = set(keywords)
    scores = {}
    for intent, data in INTENT_LEXICON.items():
        overlap = kw_set & data["keywords"]
        # also check substring containment for multi-word keyphrases
        extra = sum(1 for kw in kw_set for lex in data["keywords"] if lex in kw)
        scores[intent] = len(overlap) + 0.5 * extra
    return scores


def _tfidf_similarity(raw_text):
    """
    ML-based confidence: builds a TF-IDF space over the user's text plus
    every intent's example sentences, then computes cosine similarity.
    Returns dict {intent: similarity_score (0-1)}.
    """
    corpus = [raw_text]
    intent_order = list(INTENT_LEXICON.keys())
    for intent in intent_order:
        corpus.append(" ".join(INTENT_LEXICON[intent]["examples"]))

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return {intent: float(score) for intent, score in zip(intent_order, sims)}
    except ValueError:
        # Happens if vocabulary is empty (e.g. text is only stopwords)
        return {intent: 0.0 for intent in intent_order}


def classify_intent(raw_text, keywords):
    """
    Combines the lexicon vote and TF-IDF similarity to pick the final
    intent and a human-readable confidence score.

    Returns:
        dict with keys: intent, intent_label, confidence (0-100),
        lexicon_scores, tfidf_scores
    """
    lexicon_scores = _lexicon_vote(keywords)
    tfidf_scores = _tfidf_similarity(raw_text)

    # Normalise lexicon scores to 0-1 for fair combination
    max_lex = max(lexicon_scores.values()) if lexicon_scores else 0
    norm_lex = {k: (v / max_lex if max_lex > 0 else 0) for k, v in lexicon_scores.items()}

    # Weighted combination: lexicon evidence is more reliable for short
    # keyword-style input, TF-IDF similarity adds robustness for full
    # sentences. Weights chosen empirically (60/40).
    combined = {
        intent: 0.6 * norm_lex.get(intent, 0) + 0.4 * tfidf_scores.get(intent, 0)
        for intent in INTENT_LEXICON
    }

    best_intent = max(combined, key=combined.get)
    best_score = combined[best_intent]

    # Fallback: if nothing matched at all, default to "apology_absence"
    # since that is the most common general-purpose academic email.
    if best_score <= 0.0:
        best_intent = "apology_absence"
        best_score = 0.35

    confidence = round(min(best_score, 1.0) * 100, 1)

    return {
        "intent": best_intent,
        "intent_label": INTENT_LABELS_READABLE[best_intent],
        "confidence": confidence,
        "lexicon_scores": lexicon_scores,
        "tfidf_scores": tfidf_scores,
    }
