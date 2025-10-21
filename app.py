import io, re, datetime, urllib.parse, os
import streamlit as st
from PyPDF2 import PdfReader

# --- BASIC UTILS ---
def pdf_to_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_birads(text: str):
    m = re.search(r"BI[-\s]?RADS?\s*[:\-]?\s*(\d)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_density(text: str):
    patterns = [
        (r"extremely dense|density\s*D\b", "D"),
        (r"heterogeneously dense|density\s*C\b", "C"),
        (r"scattered fibroglandular|density\s*B\b", "B"),
        (r"almost entirely fatty|density\s*A\b", "A"),
    ]
    for pat, code in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return code
    return "unknown"

def extract_laterality(text: str):
    if re.search(r"bilateral|both breasts", text, re.IGNORECASE): return "bilateral"
    if re.search(r"\bleft breast\b", text, re.IGNORECASE): return "left"
    if re.search(r"\bright breast\b", text, re.IGNORECASE): return "right"
    return "unknown"

def timeframe_from_birads(b):
    if b is None: return None
    return {0:7, 1:365, 2:365, 3:180, 4:7, 5:7, 6:0}.get(b)

# --- LLM (optional) ---
def llm_extract(report_text: str):
    """Optional: enrich extraction with OpenAI. Returns dict or {}."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        schema = {
            "type": "object",
            "properties": {
                "birads": {"type": "integer"},
                "density": {"type": "string"},
                "laterality": {"type": "string"},
                "findings": {"type": "string"},
                "recommendation": {"type": "string"},
                "recommended_timeframe_days": {"type": "integer"}
            },
            "required": ["birads","density","laterality","findings","recommendation","recommended_timeframe_days"]
        }
        user_prompt = f"""Extract the following from this mammogram report. 
Return ONLY valid JSON matching this schema (no extra words):
{schema}

Report:
{report_text}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a strict JSON extraction engine. Output valid JSON only."},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.1
        )
        import json
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}

def patient_summary(extraction: dict, language="en"):
    # Friendly fallback summary; uses only structured fields
    b = extraction.get("birads", "not specified")
    d = extraction.get("density", "unknown")
    lat = extraction.get("laterality", "unknown")
    findings = extraction.get("findings") or "not specified in the report"
    rec = extraction.get("recommendation") or "follow your care team's advice"
    tf = extraction.get("recommended_timeframe_days")
    when = "as recommended" if tf is None else f"in about {tf} days"

    dense_line = ""
    if str(d).upper() in ["C","D"]:
        dense_line = "\n• Because your breast tissue is dense, your clinician may discuss if any additional screening is right for you."

    base = (
        "### What this means\n"
        f"• BI-RADS: **{b}**\n"
        f"• Breast density: **{d}**\n"
        f"• Side: **{lat}**\n"
        f"• Key findings: {findings}\n\n"
        "### Your next steps\n"
        f"• Recommended action: {rec} {when}.\n"
        f"{dense_line}\n"
        "### Questions to ask your clinician\n"
        "• Do I need any additional imaging?\n"
        "• When should I schedule it?\n"
        "• Is there anything else I should know?\n\n"
        "_This summary is educational and does not replace your clinician’s advice._"
    )
    return base

def calendar_link(title, start_date, details=""):
    # Create a Google Calendar "quick add" link
    # start_date = datetime.date
    start = start_date.strftime("%Y%m%d")
    end = start_date.strftime("%Y%m%d")
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{start}/{end}",
        "details": details,
    }
    return "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode(params)

# --- UI ---
st.set_page_config(page_title="Luna Breast", page_icon="🌙", layout="centered")
st.title("Luna Breast")
st.caption("Guidance for today. Confidence for tomorrow.")

st.write("Upload your mammogram report (PDF) or paste the report text. We’ll give you a warm, plain-language summary and suggested next steps.")

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Upload report PDF (optional)", type=["pdf"])
with col2:
    lang = st.selectbox("Language", ["English (en)", "Español (es)"])
    language = "es" if "Español" in lang else "en"

text_input = st.text_area("...or paste report text here", height=200, placeholder="Paste mammogram report text...")

st.divider()
use_llm = st.toggle("Use OpenAI to enhance extraction (optional)", value=False, help="If off, we use safe regex + rules only.")

if st.button("Generate my summary", type="primary"):
    if not file and not text_input.strip():
        st.error("Please upload a PDF or paste report text.")
        st.stop()

    raw_text = text_input.strip()
    if file and file.name:
        raw_text = pdf_to_text(file.read()) or raw_text

    if not raw_text.strip():
        st.error("We could not read text from the PDF. Try pasting the report text.")
        st.stop()

    # Base extraction
    birads = extract_birads(raw_text)
    density = extract_density(raw_text)
    laterality = extract_laterality(raw_text)
    timeframe = timeframe_from_birads(birads)
    extraction = {
        "birads": birads,
        "density": density,
        "laterality": laterality,
        "findings": "",
        "recommendation": "",
        "recommended_timeframe_days": timeframe,
    }

    # Optional LLM enrichment
    if use_llm:
        llm = llm_extract(raw_text)
        for k in extraction.keys():
            if k in llm and llm[k] not in [None, ""]:
                extraction[k] = llm[k]
        if extraction.get("recommended_timeframe_days") is None:
            extraction["recommended_timeframe_days"] = timeframe_from_birads(extraction.get("birads"))

    st.subheader("Your plain-language summary")
    st.markdown(patient_summary(extraction, language=language))

    # Basic readability proxy
    words = len(raw_text.split())
    sentences = max(1, raw_text.count(".")+raw_text.count("!")+raw_text.count("?"))
    avg = words / sentences
    grade = round(min(12, max(5, avg/2.0)), 1)
    st.caption(f"Approx. reading grade of the original report: {grade}")

    with st.expander("View extracted fields"):
        st.json(extraction)

    # Gentle follow-up planner (no server/background jobs needed)
    tf = extraction.get("recommended_timeframe_days")
    if tf is not None and tf >= 0:
        due = datetime.date.today() + datetime.timedelta(days=int(tf))
        st.subheader("Plan your follow-up")
        st.write(f"**Suggested target date:** {due.isoformat()}")
        cal = calendar_link("Mammogram follow-up", due, "Luna Breast reminder: schedule or complete recommended follow-up.")
        st.link_button("📅 Add to Google Calendar", cal)

    # Optional SMS via Twilio (no background jobs; just immediate send)
    with st.expander("Optional: send a quick SMS reminder now (Twilio)"):
        phone = st.text_input("Phone (E.g., +12065551234)")
        if st.button("Send SMS now"):
            sid = os.getenv("TWILIO_ACCOUNT_SID") or st.secrets.get("TWILIO_ACCOUNT_SID", None)
            tok = os.getenv("TWILIO_AUTH_TOKEN") or st.secrets.get("TWILIO_AUTH_TOKEN", None)
            frm = os.getenv("TWILIO_FROM_NUMBER") or st.secrets.get("TWILIO_FROM_NUMBER", None)
            if not (sid and tok and frm):
                st.warning("Twilio credentials not configured in Secrets, so SMS won't send.")
            else:
                try:
                    from twilio.rest import Client
                    client = Client(sid, tok)
                    msg = f"Hi from Luna Breast. Your target follow-up date is around {due.isoformat() if tf is not None else 'TBD'}."
                    client.messages.create(body=msg, from_=frm, to=phone)
                    st.success("SMS sent!")
                except Exception as e:
                    st.error(f"Twilio error: {e}")

st.write("---")
st.caption("Educational guidance only. Not medical advice.")
