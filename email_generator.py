"""
email_generator.py
-------------------
Orchestrates the full NLP pipeline:

    User Input -> Text Cleaning -> Tokenization -> Stopword Removal ->
    Lemmatization -> Keyword Extraction -> Intent Classification ->
    Entity Extraction -> Reason Classification -> Template Selection ->
    Dynamic Email Generation -> Subject Generation -> Output

Every intermediate artefact (tokens, keywords, entities, intent,
confidence, urgency) is returned alongside the final email so the
Streamlit UI can display the full pipeline for explainability during
the viva.
"""

import re

import nlp_utils as nu
import intent_classifier as ic
import templates as tpl


def _build_reason_phrase(keywords, entities, raw_text):
    """
    REASON CLASSIFICATION step.

    Converts the extracted keywords + entities into a short, natural
    reason clause that can be slotted into the email body, e.g.:
        keywords: ["cousin", "accident", "hospital"]
        -> "my cousin met with an accident and was admitted to the hospital"

    This uses simple rule-based phrase templates keyed on the presence of
    certain keyword categories, rather than free generation.
    """
    kw_set = set(keywords)
    text_l = raw_text.lower()

    # Ordered rule list: (trigger keywords, phrase). First match wins, but
    # we also try to combine complementary triggers (family + accident).
    person_words = {"father", "mother", "cousin", "brother", "sister",
                     "grandfather", "grandmother", "family", "relative"}
    person = next((p for p in person_words if p in text_l), None)

    if "hospital" in text_l or "hospitalized" in kw_set or "admitted" in kw_set:
        if person:
            return f"my {person} was hospitalized and admitted for treatment"
        return "I was hospitalized and am currently undergoing treatment"

    if "accident" in kw_set or "accident" in text_l:
        if person:
            return f"my {person} met with an accident"
        return "I met with an accident"

    if any(w in text_l for w in ("fever", "ill", "sick", "unwell", "flu", "covid")):
        return "I have been unwell and am recovering from illness"

    if "hackathon" in text_l:
        return "my team has been selected to participate in a hackathon"

    if "internship" in text_l:
        return "I have received an internship opportunity that requires my joining"

    if any(w in text_l for w in ("sports", "tournament", "match", "cricket", "football")):
        return "I have been selected to participate in a sports event/tournament"

    if any(w in text_l for w in ("nss", "camp", "blood donation", "plantation")):
        return "I am participating in an NSS activity organised by the college"

    if "industrial" in text_l and "visit" in text_l:
        return "our department has organised an industrial visit that requires my attendance"

    if any(w in text_l for w in ("competition", "hackathon", "contest", "coding")):
        return "I have been selected to participate in a technical competition"

    if any(w in text_l for w in ("club", "fest", "festival", "cultural")):
        return "I am participating in a college club/cultural activity"

    if any(w in text_l for w in ("assignment", "submission")) and "extension" in text_l or "deadline" in text_l:
        return "I faced unavoidable circumstances that affected my work"

    if any(w in text_l for w in ("thank", "thanks", "grateful", "appreciate")):
        return "your valuable guidance and support"

    if any(w in text_l for w in ("meeting", "discuss", "doubt", "appointment")):
        return "a few academic doubts that I would like to discuss with you"

    if any(w in text_l for w in ("missed", "absent", "sorry", "apolog")):
        return "I was unable to attend due to unavoidable personal reasons"

    # Generic fallback built directly from top keywords
    top_kw = [k for k in keywords if k][:3]
    if top_kw:
        return "the following reason: " + ", ".join(top_kw)
    return "an unavoidable personal reason"


def _format_date_clause(entities):
    dates = entities.get("DATE", [])
    if dates:
        return f" on {dates[0]}"
    return ""


def _format_date_suffix(entities):
    dates = entities.get("DATE", [])
    if dates:
        return f" on {dates[0]}"
    return ""


def _format_duration(entities):
    dur = entities.get("DURATION", [])
    return dur[0] if dur else ""


def generate_email(
    teacher_name: str,
    student_name: str,
    keywords_text: str,
    roll_number: str = "",
    department: str = "",
    date: str = "",
):
    """
    Runs the full pipeline end-to-end and returns a dictionary containing
    the final email plus every intermediate NLP artefact, e.g.:

    {
        "cleaned_text": ...,
        "tokens": [...],
        "tokens_no_stopwords": [...],
        "lemmas": [...],
        "keywords": [...],
        "entities": {...},
        "urgency": "High"/"Normal",
        "intent": "medical_leave",
        "intent_label": "Medical Leave",
        "confidence": 87.3,
        "reason_phrase": "...",
        "subject": "...",
        "body": "...",
    }
    """
    raw_text = keywords_text or ""

    # ---- Step 1-4: cleaning, tokenization, stopword removal, lemmatization
    cleaned = nu.clean_text(raw_text)
    tokens = nu.tokenize(cleaned)
    tokens_ns = nu.remove_stopwords(tokens)
    lemmas = nu.lemmatize(tokens_ns)

    # ---- Step 5: keyword extraction
    keywords = nu.extract_keywords(raw_text)

    # ---- Step 6: intent classification
    intent_result = ic.classify_intent(raw_text, keywords)
    intent = intent_result["intent"]

    # ---- Step 7: entity extraction
    entities = nu.extract_entities(raw_text)

    # ---- Urgency detection
    urgency = nu.detect_urgency(raw_text)

    # ---- Step 8: reason classification (build reason phrase)
    reason_phrase = _build_reason_phrase(keywords, entities, raw_text)
    reason_phrase_cap = reason_phrase[0].upper() + reason_phrase[1:] if reason_phrase else reason_phrase

    # ---- Resolve display date: explicit form field wins over extracted entity
    display_date = date.strip() if date and date.strip() else (entities["DATE"][0] if entities["DATE"] else "")
    date_clause = f" on {display_date}" if display_date else ""
    date_suffix = f" on {display_date}" if display_date else ""

    urgency_line = "This is an urgent matter and immediate attention would be appreciated. " if urgency == "High" else ""

    roll_line = f"\nRoll No: {roll_number}" if roll_number else ""
    dept_line = f"\nDepartment: {department}" if department else ""

    # ---- Step 9: template selection + Step 10: dynamic generation
    body_template = tpl.get_body_template(intent)
    body = body_template.format(
        teacher_name=nu.title_case_name(teacher_name) if teacher_name else "Sir/Madam",
        student_name=nu.title_case_name(student_name) if student_name else "Student",
        reason_phrase=reason_phrase,
        reason_phrase_cap=reason_phrase_cap,
        date_clause=date_clause,
        urgency_line=urgency_line,
        roll_line=roll_line,
        dept_line=dept_line,
    )
    body = nu.capitalize_sentences(body) if False else body  # kept explicit: template already well-formed

    # ---- Step 11: subject generation
    subject_template = tpl.get_subject_template(intent)
    try:
        subject = subject_template.format(date_suffix=date_suffix, reason_phrase=reason_phrase)
    except (KeyError, IndexError):
        subject = subject_template
    subject = subject[0].upper() + subject[1:]

    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_ns,
        "lemmas": lemmas,
        "keywords": keywords,
        "entities": entities,
        "urgency": urgency,
        "intent": intent,
        "intent_label": intent_result["intent_label"],
        "confidence": intent_result["confidence"],
        "lexicon_scores": intent_result["lexicon_scores"],
        "tfidf_scores": intent_result["tfidf_scores"],
        "reason_phrase": reason_phrase,
        "duration": _format_duration(entities),
        "subject": subject,
        "body": body,
    }
