"""
app.py
------
Streamlit front-end for the Smart Academic Email Generator.

Run with:
    streamlit run app.py
"""

import io
from datetime import date as _date

import streamlit as st

from email_generator import generate_email

# Optional PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# --------------------------------------------------------------------------
# Page config + light/dark theming
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Academic Email Generator",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "result" not in st.session_state:
    st.session_state.result = None


def inject_theme(dark: bool):
    if dark:
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #f0f2f6; }
        .email-box { background-color: #1c1f26; border: 1px solid #333; border-radius: 10px;
                     padding: 20px; color: #f0f2f6; white-space: pre-wrap; font-family: 'Georgia', serif; }
        .pipeline-chip { display:inline-block; background:#262b36; color:#7dd3fc; border-radius:14px;
                          padding:4px 12px; margin:3px; font-size:13px; border:1px solid #3a4152;}
        </style>
        """
    else:
        css = """
        <style>
        .email-box { background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px;
                     padding: 20px; color: #111; white-space: pre-wrap; font-family: 'Georgia', serif;
                     box-shadow: 0 2px 6px rgba(0,0,0,0.06);}
        .pipeline-chip { display:inline-block; background:#eef4ff; color:#1d4ed8; border-radius:14px;
                          padding:4px 12px; margin:3px; font-size:13px; border:1px solid #c7d7fb;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


inject_theme(st.session_state.dark_mode)


def chips(items):
    if not items:
        st.caption("None detected")
        return
    html = "".join(f"<span class='pipeline-chip'>{i}</span>" for i in items)
    st.markdown(html, unsafe_allow_html=True)


def make_pdf(subject: str, body: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    y = height - margin

    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Subject: " + subject)
    y -= 12 * mm

    c.setFont("Helvetica", 11)
    for paragraph in body.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for w in words:
            trial = (line + " " + w).strip()
            if c.stringWidth(trial, "Helvetica", 11) > (width - 2 * margin):
                c.drawString(margin, y, line)
                y -= 6 * mm
                line = w
                if y < margin:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = height - margin
            else:
                line = trial
        c.drawString(margin, y, line)
        y -= 6 * mm
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - margin

    c.save()
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------
# Sidebar: input form
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("✉️ Email Details")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

    teacher_name = st.text_input("Teacher Name", placeholder="e.g. Dr. Patil")
    student_name = st.text_input("Student Name", placeholder="e.g. Paras Nalte")
    roll_number = st.text_input("Roll Number (optional)", placeholder="e.g. 21")
    department = st.text_input("Department (optional)", placeholder="e.g. AI & Data Science")
    event_date = st.date_input("Date", value=_date.today())
    include_date = st.checkbox("Include this date in the email", value=True)

    keywords_text = st.text_area(
        "Keyword Input Box",
        height=160,
        placeholder=(
            "Type short keywords or phrases, one idea per line, e.g.\n"
            "cousin accident\nhospital\nfather in office\nurgent\nmissed lab\n21 September"
        ),
    )

    col_a, col_b = st.columns(2)
    generate_clicked = col_a.button("🚀 Generate Email", use_container_width=True, type="primary")
    clear_clicked = col_b.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.result = None
    st.rerun()

if generate_clicked:
    if not teacher_name or not student_name or not keywords_text.strip():
        st.sidebar.error("Please fill Teacher Name, Student Name and Keywords.")
    else:
        date_str = event_date.strftime("%d %B %Y") if include_date else ""
        st.session_state.result = generate_email(
            teacher_name=teacher_name,
            student_name=student_name,
            keywords_text=keywords_text,
            roll_number=roll_number,
            department=department,
            date=date_str,
        )

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.markdown("## 🎓 Smart Academic Email Generator")
st.caption("Converts short keywords into a complete, professional academic email — powered by classical NLP, not an LLM.")

result = st.session_state.result

if result is None:
    st.info("Fill in the details in the sidebar and click **Generate Email** to get started.")
else:
    tab_email, tab_pipeline, tab_edit = st.tabs(["📧 Generated Email", "🔬 NLP Pipeline", "✏️ Edit Email"])

    with tab_email:
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected Intent", result["intent_label"])
        c2.metric("Confidence Score", f"{result['confidence']}%")
        c3.metric("Urgency", result["urgency"])

        st.markdown("#### Subject")
        subject_display = st.session_state.get("edited_subject", result["subject"])
        st.code(subject_display, language=None)

        st.markdown("#### Body")
        body_display = st.session_state.get("edited_body", result["body"])
        st.markdown(f"<div class='email-box'>{body_display}</div>", unsafe_allow_html=True)

        full_email_text = f"Subject: {subject_display}\n\n{body_display}"

        st.markdown("##### Export")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "⬇️ Download as TXT",
                data=full_email_text,
                file_name="academic_email.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with e2:
            if REPORTLAB_AVAILABLE:
                pdf_bytes = make_pdf(subject_display, body_display)
                st.download_button(
                    "⬇️ Download as PDF",
                    data=pdf_bytes,
                    file_name="academic_email.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.button("PDF export unavailable (install reportlab)", disabled=True, use_container_width=True)
        with e3:
            st.text_area("📋 Copy from here", value=full_email_text, height=1, label_visibility="collapsed")
            st.caption("Select all text above (Ctrl/Cmd+A) and copy (Ctrl/Cmd+C).")

    with tab_pipeline:
        st.markdown("Each stage below is a **classical NLP technique** (no LLM is used to write the email).")

        st.markdown("**1. Text Cleaning**")
        st.code(result["cleaned_text"] or "(empty)")

        st.markdown("**2. Tokenization**")
        chips(result["tokens"])

        st.markdown("**3. Stopword Removal**")
        chips(result["tokens_no_stopwords"])

        st.markdown("**4. Lemmatization**")
        chips(result["lemmas"])

        st.markdown("**5. Keyword Extraction**")
        chips(result["keywords"])

        st.markdown("**6. Intent Classification**")
        st.write(f"Predicted intent: **{result['intent_label']}**  |  Confidence: **{result['confidence']}%**")
        with st.expander("Show lexicon vote scores + TF-IDF cosine similarity scores"):
            cols = st.columns(2)
            cols[0].write("Lexicon (rule-based) scores")
            cols[0].json(result["lexicon_scores"])
            cols[1].write("TF-IDF cosine similarity scores (scikit-learn)")
            cols[1].json({k: round(v, 3) for k, v in result["tfidf_scores"].items()})

        st.markdown("**7. Named Entity Recognition (spaCy)**")
        st.json(result["entities"])

        st.markdown("**8. Urgency Detection**")
        st.write(f"Urgency level: **{result['urgency']}**")

        st.markdown("**9. Reason Classification**")
        st.write(result["reason_phrase"])

        st.markdown("**10-11. Template Selection & Dynamic Generation**")
        st.write(f"Selected template family: `{result['intent']}`")

    with tab_edit:
        st.markdown("Fine-tune the generated email before sending.")
        edited_subject = st.text_input("Edit Subject", value=result["subject"], key="edited_subject")
        edited_body = st.text_area("Edit Body", value=result["body"], height=350, key="edited_body")
        st.success("Edits are reflected live in the **Generated Email** tab.")

st.markdown("---")
st.caption("Smart Academic Email Generator · NLP Mini Project · Classical NLP pipeline (spaCy, NLTK, scikit-learn) — no LLM/OpenAI API used.")
