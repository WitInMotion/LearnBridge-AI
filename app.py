import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

import streamlit as st


APP_NAME = "LearnBridge AI"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_OPTIONS = {
    "Gemma 4 (e2b)": "gemma4:e2b",
    "Gemma 3 (1b)": "gemma3:1b",
}
SUBJECTS = ["Mathematics", "English", "Biology", "Chemistry", "Physics", "Economics"]
EXAM_MODES = ["WAEC", "JAMB", "General Study"]


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "page": "chat",
        "messages": [],
        "quiz_state": {"idx": 0, "score": 0, "answered": False, "selected": None},
        "subject": "Mathematics",
        "exam": "WAEC",
        "model_label": "Gemma 4 (e2b)",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.messages:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am LearnBridge AI. Ask me anything from your WAEC or JAMB study plan, "
                    "and I will explain it clearly with steps, examples, and a quick practice check."
                ),
                "time": datetime.now().strftime("%I:%M %p"),
            }
        ]


def handle_query_params() -> None:
    params = st.query_params
    changed = False

    if "page" in params and params["page"] in {"chat", "quiz"}:
        st.session_state.page = params["page"]
        changed = True

    if "action" in params:
        action_prompts = {
            "simple": "Explain quadratic equations simply with one worked example.",
            "smarter": "How can I study smarter with AI for WAEC and JAMB?",
            "improve": "Help me create a simple progress plan for exam revision.",
            "practice": None,
        }
        action = params["action"]
        if action == "practice":
            st.session_state.page = "quiz"
        elif action in action_prompts:
            st.session_state.page = "chat"
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": action_prompts[action],
                    "time": datetime.now().strftime("%I:%M %p"),
                }
            )
        changed = True

    if changed:
        st.query_params.clear()
        time.sleep(0.05)
        st.rerun()


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def build_system_prompt(subject: str, exam: str) -> str:
    return f"""
You are LearnBridge AI, a friendly but accurate tutor for Nigerian teenagers and adult learners preparing for {exam}.
The selected subject is {subject}.

Rules:
- Give the direct answer first when there is a clear answer.
- Explain in simple language without talking down to the learner.
- Show steps for calculations and reasoning questions.
- Use WAEC/JAMB style where relevant.
- If the question is ambiguous, state the likely interpretation and answer it.
- End with one short practice question or quick check.
- If you are unsure, say what extra information is needed instead of inventing facts.
""".strip()


def ollama_chat(prompt: str, subject: str, exam: str, model_id: str) -> str:
    history = []
    for msg in st.session_state.messages[-8:]:
        if msg["role"] in {"user", "assistant"}:
            history.append({"role": msg["role"], "content": strip_html(msg["content"])})

    payload = {
        "model": model_id,
        "stream": False,
        "messages": [
            {"role": "system", "content": build_system_prompt(subject, exam)},
            *history,
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.35,
            "top_p": 0.9,
            "num_predict": 700,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return (
            "I could not reach Ollama at `localhost:11434`. Start Ollama and make sure the selected "
            f"model `{model_id}` is available, then ask again.\n\nDetails: {exc}"
        )
    except TimeoutError:
        return "The model took too long to answer. Try the lighter `gemma3:1b` model or ask a shorter question."
    except json.JSONDecodeError:
        return "Ollama responded, but the response was not valid JSON. Please try again."

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        return "The model returned an empty response. Please try again with a little more detail."
    return content


def answer_quiz_with_ai(question: str, selected: str, correct: str, subject: str, exam: str, model_id: str) -> str:
    prompt = (
        f"Explain this {exam} {subject} quiz answer briefly.\n"
        f"Question: {question}\n"
        f"Student selected: {selected}\n"
        f"Correct answer: {correct}\n"
        "Explain why the correct option is right in 2-4 sentences."
    )
    return ollama_chat(prompt, subject, exam, model_id)


def markdown_message(text: str) -> None:
    st.markdown(text.replace("\n", "  \n"))


CUSTOM_CSS = """
<style>
:root {
    --bg-main: #070b13;
    --bg-sidebar: #080a12;
    --panel: #101827;
    --panel-soft: #141d2f;
    --line: rgba(148, 163, 184, 0.22);
    --text: #f8fafc;
    --muted: #a7b2c5;
    --blue: #2563eb;
    --cyan: #06b6d4;
    --violet: #7c3aed;
    --pink: #db2777;
    --green: #10b981;
    --amber: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 95% 0%, rgba(219, 39, 119, 0.18), transparent 30rem),
        radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.16), transparent 26rem),
        var(--bg-main) !important;
    color: var(--text);
}

#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
div.block-container {
    max-width: 1320px;
    padding: 1.5rem 1.25rem 4.5rem;
}

section[data-testid="stSidebar"] {
    width: 340px !important;
    background: linear-gradient(180deg, #090b14 0%, #071120 100%) !important;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div { padding: 1rem 1.2rem; }

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 6px 34px;
}
.sidebar-logo {
    width: 52px;
    height: 52px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    box-shadow: 0 16px 40px rgba(37, 99, 235, 0.24);
    font-size: 28px;
}
.brand-title { font-size: 23px; font-weight: 800; letter-spacing: 0; }
.brand-sub { color: var(--muted); font-size: 13px; margin-top: 3px; }

.nav-card {
    display: block;
    color: var(--text) !important;
    text-decoration: none !important;
    padding: 18px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(20, 29, 47, 0.62);
    margin-bottom: 14px;
    transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
}
.nav-card:hover {
    transform: translateY(-1px);
    background: rgba(30, 41, 59, 0.72);
    border-color: rgba(125, 211, 252, 0.34);
}
.nav-card.active {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.58), rgba(124, 58, 237, 0.5));
    border-color: rgba(167, 139, 250, 0.62);
}
.nav-title { display: flex; align-items: center; gap: 12px; font-weight: 750; }
.nav-sub { color: #d5dbea; font-size: 12px; margin: 7px 0 0 34px; }

.section-title {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.12em;
    font-weight: 800;
    margin: 32px 0 16px;
}
.field-label {
    color: var(--text);
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 14px;
    font-weight: 700;
    margin: 18px 0 8px;
}
div[data-testid="stSidebar"] .stSelectbox label { display: none; }
div[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.88) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    min-height: 48px;
}

.model-note {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 9px;
    border-radius: 999px;
    color: #8ef3c5;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    font-size: 11px;
    font-weight: 800;
}
.success-card {
    margin-top: 38px;
    padding: 22px;
    border-radius: 8px;
    background: linear-gradient(145deg, rgba(37, 99, 235, 0.16), rgba(124, 58, 237, 0.22));
    border: 1px solid rgba(167, 139, 250, 0.36);
}
.success-title { color: #f0abfc; font-weight: 800; margin-bottom: 14px; }
.success-text { color: #dbe5f5; line-height: 1.75; font-size: 14px; }

.hero-banner {
    min-height: 300px;
    border-radius: 8px;
    padding: 48px 40px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(125, 211, 252, 0.24);
    background:
        radial-gradient(circle at 102% -6%, rgba(244, 63, 94, 0.54), transparent 15rem),
        radial-gradient(circle at 7% 110%, rgba(67, 56, 202, 0.5), transparent 18rem),
        linear-gradient(120deg, #063d89 0%, #231078 42%, #831c86 74%, #c0266f 100%);
    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.32);
}
.hero-banner:before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(circle, rgba(255,255,255,0.15) 1px, transparent 1px),
        linear-gradient(115deg, transparent 0 38%, rgba(255,255,255,0.06) 38% 39%, transparent 39%);
    background-size: 52px 52px, 100% 100%;
    opacity: 0.34;
}
.hero-shape-left,
.hero-shape-right {
    position: absolute;
    opacity: 0.22;
    filter: blur(0.2px);
    pointer-events: none;
}
.hero-shape-left { left: 90px; top: 142px; font-size: 58px; transform: rotate(-16deg); }
.hero-shape-right { right: 130px; top: 138px; font-size: 58px; transform: rotate(17deg); }
.hero-title {
    position: relative;
    z-index: 1;
    text-align: center;
    font-size: clamp(38px, 5vw, 56px);
    line-height: 1.02;
    font-weight: 850;
    margin: 0;
    letter-spacing: 0;
}
.hero-underline {
    width: min(340px, 56vw);
    height: 6px;
    border-radius: 999px;
    margin: 16px auto 18px;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f59e0b);
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    position: relative;
    z-index: 1;
    text-align: center;
    color: #edf4ff;
    font-size: 18px;
    margin: 0;
}
.features-row {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 24px;
    max-width: 830px;
    margin: 42px auto 0;
}
.feature {
    text-decoration: none !important;
    color: var(--text) !important;
    text-align: center;
}
.feature-icon {
    width: 56px;
    height: 56px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    margin: 0 auto 14px;
    font-size: 24px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.24);
}
.feature-text { font-size: 15px; line-height: 1.45; color: #f4f7fb; }
.blue { background: linear-gradient(135deg, #0ea5e9, #2563eb); }
.green { background: linear-gradient(135deg, #22c55e, #0f9f78); }
.violet { background: linear-gradient(135deg, #a855f7, #6d28d9); }
.amber { background: linear-gradient(135deg, #fbbf24, #f97316); }

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(10, 18, 32, 0.82);
    overflow: hidden;
}
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 26px;
    border-bottom: 1px solid var(--line);
    background: rgba(16, 24, 39, 0.82);
}
.panel-title-wrap { display: flex; gap: 16px; align-items: center; }
.panel-icon {
    width: 52px;
    height: 52px;
    display: grid;
    place-items: center;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.34), rgba(124, 58, 237, 0.38));
    color: #d8b4fe;
    font-size: 25px;
}
.panel-title { font-size: 23px; font-weight: 800; }
.panel-sub { color: var(--muted); font-size: 14px; margin-top: 5px; }
.panel-body { padding: 26px; min-height: 390px; }

.stButton > button {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    background: rgba(15, 23, 42, 0.8) !important;
    color: var(--text) !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: rgba(125, 211, 252, 0.45) !important;
    background: rgba(30, 41, 59, 0.9) !important;
}

div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    margin: 0 0 20px !important;
}
div[data-testid="stChatMessageContent"] {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 8px;
    padding: 16px 18px;
    background: rgba(20, 29, 47, 0.82);
    color: var(--text) !important;
    line-height: 1.68;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #6d28d9, #db2777);
    border-color: rgba(244, 114, 182, 0.36);
    margin-left: auto;
    max-width: min(760px, 86%);
}
.msg-time { color: rgba(255,255,255,0.72); font-size: 11px; text-align: right; margin-top: 8px; }

div[data-testid="stChatInput"] > div {
    border-radius: 8px !important;
    background: #111827 !important;
    border: 1px solid var(--line) !important;
}
div[data-testid="stChatInput"] textarea { color: var(--text) !important; }

.quiz-question {
    font-size: 23px;
    font-weight: 800;
    line-height: 1.35;
    margin-bottom: 22px;
}
.quiz-meta { color: var(--muted); margin-bottom: 18px; }
.answer-card {
    padding: 16px 18px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(15, 23, 42, 0.78);
    margin-top: 12px;
}
.footer-bar {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 100;
    height: 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 30px;
    background: rgba(6, 10, 18, 0.96);
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 12px;
}

@media (max-width: 820px) {
    div.block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    .hero-banner { padding: 34px 18px; }
    .features-row { grid-template-columns: repeat(2, 1fr); gap: 22px 12px; }
    .panel-header { align-items: flex-start; gap: 16px; flex-direction: column; }
    .footer-bar { display: none; }
}
</style>
"""


QUIZ_POOL = {
    "Mathematics": [
        {
            "q": "Solve for x: 2x + 5 = 15",
            "options": ["x = 3", "x = 5", "x = 10", "x = 20"],
            "ans": "x = 5",
        },
        {
            "q": "What is the value of 3² + 4²?",
            "options": ["7", "12", "25", "49"],
            "ans": "25",
        },
        {
            "q": "The gradient of a straight line is also called its...",
            "options": ["Area", "Slope", "Volume", "Intercept"],
            "ans": "Slope",
        },
    ],
    "English": [
        {
            "q": "Choose the correctly punctuated sentence.",
            "options": [
                "I went to Lagos, and bought books.",
                "I went to Lagos and bought books.",
                "I went, to Lagos and bought books.",
                "I went to Lagos and, bought books.",
            ],
            "ans": "I went to Lagos and bought books.",
        },
        {
            "q": "Which word is closest in meaning to 'brief'?",
            "options": ["Short", "Heavy", "Noisy", "Late"],
            "ans": "Short",
        },
    ],
    "Biology": [
        {
            "q": "Which organelle is mainly responsible for photosynthesis?",
            "options": ["Nucleus", "Chloroplast", "Ribosome", "Mitochondrion"],
            "ans": "Chloroplast",
        },
        {
            "q": "The basic unit of life is the...",
            "options": ["Tissue", "Organ", "Cell", "System"],
            "ans": "Cell",
        },
    ],
    "Chemistry": [
        {
            "q": "What is the chemical symbol for sodium?",
            "options": ["S", "So", "Na", "N"],
            "ans": "Na",
        },
        {
            "q": "A substance with pH less than 7 is...",
            "options": ["Acidic", "Neutral", "Alkaline", "Metallic"],
            "ans": "Acidic",
        },
    ],
    "Physics": [
        {
            "q": "The SI unit of force is...",
            "options": ["Joule", "Watt", "Newton", "Pascal"],
            "ans": "Newton",
        },
        {
            "q": "Speed is calculated as...",
            "options": ["Distance / Time", "Time / Distance", "Mass x Acceleration", "Force / Area"],
            "ans": "Distance / Time",
        },
    ],
    "Economics": [
        {
            "q": "Demand usually rises when price...",
            "options": ["Rises", "Falls", "Stays impossible", "Becomes tax"],
            "ans": "Falls",
        },
        {
            "q": "The reward for labour is called...",
            "options": ["Rent", "Interest", "Wages", "Profit"],
            "ans": "Wages",
        },
    ],
}


def render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">🎓</div>
                <div>
                    <div class="brand-title">LearnBridge AI</div>
                    <div class="brand-sub">Learn Smarter. Score Higher.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_active = "active" if st.session_state.page == "chat" else ""
        quiz_active = "active" if st.session_state.page == "quiz" else ""
        st.markdown(
            f"""
            <a href="/?page=chat" target="_self" class="nav-card {chat_active}">
                <div class="nav-title"><span>💬</span><span>Chat Tutor</span></div>
                <div class="nav-sub">Ask anything, get clear explanations</div>
            </a>
            <a href="/?page=quiz" target="_self" class="nav-card {quiz_active}">
                <div class="nav-title"><span>📝</span><span>Practice Quiz</span></div>
                <div class="nav-sub">Test your knowledge with quizzes</div>
            </a>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Learning Settings</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">📚 Subject</div>', unsafe_allow_html=True)
        subject = st.selectbox(
            "Subject",
            SUBJECTS,
            index=SUBJECTS.index(st.session_state.subject),
            label_visibility="collapsed",
            key="subject_select",
        )
        st.session_state.subject = subject

        st.markdown('<div class="field-label">🎯 Exam Mode</div>', unsafe_allow_html=True)
        exam = st.selectbox(
            "Exam Mode",
            EXAM_MODES,
            index=EXAM_MODES.index(st.session_state.exam),
            label_visibility="collapsed",
            key="exam_select",
        )
        st.session_state.exam = exam

        st.markdown('<div class="field-label">🤖 AI Model</div>', unsafe_allow_html=True)
        model_label = st.selectbox(
            "AI Model",
            list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.model_label),
            label_visibility="collapsed",
            key="model_select",
        )
        st.session_state.model_label = model_label
        st.markdown(f'<span class="model-note">{MODEL_OPTIONS[model_label]}</span>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="success-card">
                <div class="success-title">Your success is possible</div>
                <div class="success-text">
                    Each question you ask is practice. Keep it steady, keep it honest, and keep moving.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return subject, exam, MODEL_OPTIONS[model_label]


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-shape-left">📚</div>
            <div class="hero-shape-right">🏆</div>
            <h1 class="hero-title">Welcome to LearnBridge AI</h1>
            <div class="hero-underline"></div>
            <p class="hero-subtitle">Your personal AI tutor for WAEC & JAMB success</p>
            <div class="features-row">
                <a href="/?action=simple" target="_self" class="feature">
                    <div class="feature-icon blue">💬</div>
                    <div class="feature-text">Get simple<br>explanations</div>
                </a>
                <a href="/?action=practice" target="_self" class="feature">
                    <div class="feature-icon green">🎯</div>
                    <div class="feature-text">Practice with<br>exam-style questions</div>
                </a>
                <a href="/?action=smarter" target="_self" class="feature">
                    <div class="feature-icon violet">🧠</div>
                    <div class="feature-text">Learn smarter<br>with AI</div>
                </a>
                <a href="/?action=improve" target="_self" class="feature">
                    <div class="feature-icon amber">📈</div>
                    <div class="feature-text">Improve and track<br>your progress</div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat(subject: str, exam: str, model_id: str) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-header">
            <div class="panel-title-wrap">
                <div class="panel-icon">💬</div>
                <div>
                    <div class="panel-title">Chat with LearnBridge AI</div>
                    <div class="panel-sub">Ask any question and get clear, step-by-step explanations.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clear_col, model_col = st.columns([1, 4])
    with clear_col:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with model_col:
        st.caption(f"Using `{model_id}` for {exam} {subject}")

    st.markdown('<div class="panel-body">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(
                    f'{msg["content"]}<div class="msg-time">{msg.get("time", "")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                markdown_message(msg["content"])
    st.markdown("</div></div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "time": datetime.now().strftime("%I:%M %p")}
        )
        with st.spinner(f"Asking {model_id}..."):
            response = ollama_chat(prompt, subject, exam, model_id)
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "time": datetime.now().strftime("%I:%M %p")}
        )
        st.rerun()


def render_quiz(subject: str, exam: str, model_id: str) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-header">
            <div class="panel-title-wrap">
                <div class="panel-icon">📝</div>
                <div>
                    <div class="panel-title">Practice Quiz</div>
                    <div class="panel-sub">Answer exam-style questions, then ask the model to explain the result.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-body">', unsafe_allow_html=True)
    pool = QUIZ_POOL.get(subject, QUIZ_POOL["Mathematics"])
    q_state = st.session_state.quiz_state
    current_q = pool[q_state["idx"] % len(pool)]

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(
            f'<div class="quiz-meta">{exam} • {subject} • Question {q_state["idx"] + 1}</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Reset Quiz", use_container_width=True):
            st.session_state.quiz_state = {"idx": 0, "score": 0, "answered": False, "selected": None}
            st.rerun()

    st.markdown(f'<div class="quiz-question">{current_q["q"]}</div>', unsafe_allow_html=True)

    if not q_state["answered"]:
        for option in current_q["options"]:
            if st.button(option, use_container_width=True, key=f"option_{q_state['idx']}_{option}"):
                st.session_state.quiz_state["selected"] = option
                st.session_state.quiz_state["answered"] = True
                if option == current_q["ans"]:
                    st.session_state.quiz_state["score"] += 1
                st.rerun()
    else:
        selected = q_state["selected"]
        correct = current_q["ans"]
        if selected == correct:
            st.success("Correct. Nice work.")
        else:
            st.error(f"Not quite. The correct answer is: {correct}")

        with st.expander("AI explanation", expanded=True):
            explanation_key = f"explanation_{q_state['idx']}_{selected}"
            if explanation_key not in st.session_state:
                with st.spinner(f"Asking {model_id} to explain..."):
                    st.session_state[explanation_key] = answer_quiz_with_ai(
                        current_q["q"], selected, correct, subject, exam, model_id
                    )
            markdown_message(st.session_state[explanation_key])

        next_col, score_col = st.columns([1, 3])
        with next_col:
            if st.button("Next Question", use_container_width=True):
                st.session_state.quiz_state["idx"] += 1
                st.session_state.quiz_state["answered"] = False
                st.session_state.quiz_state["selected"] = None
                st.rerun()
        with score_col:
            st.markdown(f'<div class="answer-card">Score: {q_state["score"]}</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-bar">
            <div>Built with care using Gemma, Streamlit and Ollama</div>
            <div>Empowering students. Building brighter futures.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_state()
handle_query_params()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
selected_subject, selected_exam, selected_model = render_sidebar()
render_hero()

if st.session_state.page == "quiz":
    render_quiz(selected_subject, selected_exam, selected_model)
else:
    render_chat(selected_subject, selected_exam, selected_model)

render_footer()
