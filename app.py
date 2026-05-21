# import warnings
# warnings.filterwarnings("ignore")
# import time
# import streamlit as st
# from engine import answer, classify_query
# from database import init_db
# from document import build_index
# import os

# # Page config
# st.set_page_config(
#     page_title="CliniqAI",
#     page_icon="⚕️",
#     layout="wide"
# )

# # Initialize on first run
# @st.cache_resource
# @st.cache_resource
# def initialize():
#     import os
#     db_path = os.path.join(os.path.dirname(__file__), "cliniqai.db")
#     index_path = os.path.join(os.path.dirname(__file__), "cliniqai.index")
    
#     if not os.path.exists(db_path):
#         init_db()
#     if not os.path.exists(index_path):
#         build_index()
#     return True

# initialize()

# # Header
# col1, col2 = st.columns([3, 1])
# with col1:
#     st.title("⚕️ CliniqAI")
#     st.caption("Intelligent Clinical Assistant: Powered by MIMIC-III & Groq")
# with col2:
#     st.metric("Patients", "100")
#     st.metric("Data Source", "MIMIC-III")

# st.divider()

# # Sidebar
# with st.sidebar:
#     st.header("About CliniqAI")
#     st.markdown("""
#     CliniqAI unifies fragmented healthcare data into a
#     single natural language interface.

#     **Data Sources:**
#     - 100 de-identified ICU patients
#     - Real admission & diagnosis records
#     - Prescription data
#     - MIMIC-III Clinical Database

#     **Architecture:**
#     - 🔤 Single-token LLM Router
#     - 🗄️ SQL — structured queries
#     - 🧠 RAG — semantic search (FAISS)
#     - ⚡ HYBRID — parallel SQL + RAG
#     - 💬 DIRECT — conversational
#     """)

#     st.divider()

#     st.header("Try These Questions")
#     example_questions = [
#         "How many patients were admitted as emergency?",
#         "Show me patients with cardiac conditions",
#         "What medications are most commonly prescribed?",
#         "How many female patients do we have?",
#         "Which patients died during admission?",
#         "What is the breakdown of admission types?",
#     ]
#     for q in example_questions:
#         if st.button(q, use_container_width=True):
#             st.session_state.pending_question = q

#     st.divider()

#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.messages = []
#         st.rerun()

#     st.divider()
#     st.caption("LLM: Llama 3.3 70B via Groq")
#     st.caption("Embeddings: all-MiniLM-L6-v2")
#     st.caption("Vector Store: FAISS")

# # Route badge styling
# ROUTE_CONFIG = {
#     "S": {"label": "SQL", "color": "#1f77b4", "icon": "🗄️"},
#     "R": {"label": "RAG", "color": "#2ca02c", "icon": "🧠"},
#     "H": {"label": "HYBRID", "color": "#9467bd", "icon": "⚡"},
#     "D": {"label": "DIRECT", "color": "#7f7f7f", "icon": "💬"},
# }

# def render_route_badge(route):
#     config = ROUTE_CONFIG.get(route, ROUTE_CONFIG["D"])
#     st.markdown(
#         f"""<span style='
#             background-color: {config["color"]}22;
#             color: {config["color"]};
#             border: 1px solid {config["color"]};
#             padding: 2px 10px;
#             border-radius: 12px;
#             font-size: 12px;
#             font-weight: bold;
#         '>{config["icon"]} {config["label"]}</span>""",
#         unsafe_allow_html=True
#     )

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "pending_question" not in st.session_state:
#     st.session_state.pending_question = None

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message.get("route"):
#             render_route_badge(message["route"])
#         if message.get("sql"):
#             with st.expander("🔍 SQL Query Generated"):
#                 st.code(message["sql"], language="sql")

# def build_chat_history():
#     return [
#         {"role": m["role"], "content": m["content"]}
#         for m in st.session_state.messages[-20:]
#     ]

# def process_question(question):
#     # Add user message
#     st.session_state.messages.append({
#         "role": "user",
#         "content": question
#     })

#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         # Thinking states
#         status = st.empty()

#         status.markdown("🔄 **Classifying intent...**")
#         time.sleep(0.3)

#         route = classify_query(question)
#         config = ROUTE_CONFIG.get(route.upper(), ROUTE_CONFIG["D"])

#         if route == "s":
#             status.markdown(f"🗄️ **Querying SQL database...**")
#         elif route == "r":
#             status.markdown(f"🧠 **Searching FAISS vector space...**")
#         elif route == "h":
#             status.markdown(f"⚡ **Running parallel SQL + RAG retrieval...**")
#         else:
#             status.markdown(f"💬 **Thinking...**")

#         chat_history = build_chat_history()
#         result = answer(question, chat_history)

#         status.markdown("✍️ **Synthesizing clinical answer...**")
#         time.sleep(0.3)
#         status.empty()

#         # Render answer
#         st.markdown(result["answer"])
#         render_route_badge(result["route"])

#         if result["sql_used"]:
#             with st.expander("🔍 SQL Query Generated"):
#                 st.code(result["sql_used"], language="sql")

#     # Save to history
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "route": result["route"],
#         "sql": result["sql_used"]
#     })

#     st.rerun()

# # Handle sidebar button clicks
# if st.session_state.pending_question:
#     question = st.session_state.pending_question
#     st.session_state.pending_question = None
#     process_question(question)

# # Chat input
# if question := st.chat_input("Ask anything about your patients..."):
#     process_question(question)

import warnings
warnings.filterwarnings("ignore")
import time
import streamlit as st
from engine import answer, classify_query
from database import init_db
from document import build_index
import os

st.set_page_config(
    page_title="CliniqAI",
    page_icon="⚕️",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.stApp { background-color: #07132A; }
.block-container { padding-top: 1rem !important; padding-left: 2rem; padding-right: 2rem; }
[data-testid="stSidebar"] { background-color: #0A1F3D; border-right: 1px solid #183560; }
[data-testid="stSidebar"] * { color: #EFF6FF; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: #7A9CC4 !important; font-size: 12px; line-height: 1.7; }
[data-testid="stSidebar"] .stButton > button { background-color: #0D2040; color: #7A9CC4 !important; border: 1px solid #183560; border-radius: 8px; font-size: 11px; padding: 6px 10px; width: 100%; transition: all 0.15s; }
[data-testid="stSidebar"] .stButton > button:hover { border-color: #1AC8D4; color: #1AC8D4 !important; background-color: #041820; }
h1, h2, h3, p, span, label, li { color: #EFF6FF !important; }
.stCaption, small { color: #6B9CC4 !important; }
hr { border-color: #183560 !important; margin: 0.8rem 0; }
[data-testid="stMetric"] { background-color: #0D2040; border: 1px solid #183560; border-radius: 10px; padding: 10px 14px; }
[data-testid="stMetricLabel"] { color: #6B9CC4 !important; font-size: 10px !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #1AC8D4 !important; font-size: 1.4rem !important; font-weight: 600 !important; }
[data-testid="stChatMessage"] { background-color: #0D2040; border: 1px solid #183560; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span { color: #EFF6FF !important; }
[data-testid="stChatInput"] { background-color: #0D2040 !important; border: 1.5px solid #1AC8D4 !important; border-radius: 12px !important; }
[data-testid="stChatInput"] textarea { color: #EFF6FF !important; background-color: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #3D5A7A !important; }
[data-testid="stChatInput"] button { background-color: #1AC8D4 !important; border-radius: 8px !important; }
[data-testid="stChatInput"] button svg { fill: #07132A !important; }
[data-testid="stExpander"] { background-color: #07132A; border: 1px solid #183560; border-radius: 8px; }
[data-testid="stExpander"] summary { color: #6B9CC4 !important; font-size: 12px; }
[data-testid="stExpander"] summary:hover { color: #1AC8D4 !important; }
.stCodeBlock, pre { background-color: #030C1A !important; border: 1px solid #183560 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize():
    db_path = os.path.join(os.path.dirname(__file__), "cliniqai.db")
    index_path = os.path.join(os.path.dirname(__file__), "cliniqai.index")
    if not os.path.exists(db_path):
        init_db()
    if not os.path.exists(index_path):
        build_index()
    return True

initialize()

# Header
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.title("⚕️ CliniqAI")
    st.caption("Intelligent Clinical Assistant: Powered by MIMIC-III & Groq")
with col2:
    st.metric("Patients", "100")
with col3:
    st.metric("Data Source", "MIMIC-III")

st.divider()

# Sidebar
with st.sidebar:
    st.header("About CliniqAI")
    st.markdown("""
    CliniqAI unifies fragmented healthcare data into a
    single natural language interface.

    **Data Sources:**
    - 100 de-identified ICU patients
    - Real admission & diagnosis records
    - Prescription data
    - MIMIC-III Clinical Database

    **Architecture:**
    - 🔤 Single-token LLM Router
    - 🗄️ SQL — structured queries
    - 🧠 RAG — semantic search (FAISS)
    - ⚡ HYBRID — parallel SQL + RAG
    - 💬 DIRECT — conversational
    """)

    st.divider()

    st.header("Try These Questions")
    example_questions = [
        "How many patients were admitted as emergency?",
        "Show me patients with cardiac conditions",
        "What medications are most commonly prescribed?",
        "How many female patients do we have?",
        "Which patients died during admission?",
        "What is the breakdown of admission types?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("LLM: Llama 3.3 70B via Groq")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector Store: FAISS")

# Route badge styling
ROUTE_CONFIG = {
    "S": {"label": "SQL",    "color": "#0EA5E9", "icon": "🗄️"},
    "R": {"label": "RAG",    "color": "#1AC8D4", "icon": "🧠"},
    "H": {"label": "HYBRID", "color": "#6366F1", "icon": "⚡"},
    "D": {"label": "DIRECT", "color": "#6B9CC4", "icon": "💬"},
}

def render_route_badge(route):
    config = ROUTE_CONFIG.get(route, ROUTE_CONFIG["D"])
    st.markdown(
        f"""<span style='background-color:{config["color"]}22;color:{config["color"]};border:1px solid {config["color"]};padding:2px 10px;border-radius:12px;font-size:11px;font-weight:bold;'>{config["icon"]} {config["label"]}</span>""",
        unsafe_allow_html=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            render_route_badge(message["route"])
        if message.get("sql"):
            with st.expander("🔍 SQL Query Generated"):
                st.code(message["sql"], language="sql")

def build_chat_history():
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-20:]
    ]

def process_question(question):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("<span style='color:#6B9CC4;font-size:12px;'>🔄 Classifying intent...</span>", unsafe_allow_html=True)
        time.sleep(0.3)

        route = classify_query(question)

        if route == "s":
            status.markdown("<span style='color:#0EA5E9;font-size:12px;'>🗄️ Querying SQL database...</span>", unsafe_allow_html=True)
        elif route == "r":
            status.markdown("<span style='color:#1AC8D4;font-size:12px;'>🧠 Searching FAISS vector space...</span>", unsafe_allow_html=True)
        elif route == "h":
            status.markdown("<span style='color:#6366F1;font-size:12px;'>⚡ Running parallel SQL + RAG retrieval...</span>", unsafe_allow_html=True)
        else:
            status.markdown("<span style='color:#6B9CC4;font-size:12px;'>💬 Thinking...</span>", unsafe_allow_html=True)

        chat_history = build_chat_history()
        result = answer(question, chat_history)

        status.markdown("<span style='color:#1AC8D4;font-size:12px;'>✍️ Synthesizing clinical answer...</span>", unsafe_allow_html=True)
        time.sleep(0.3)
        status.empty()

        st.markdown(result["answer"])
        render_route_badge(result["route"])

        if result["sql_used"]:
            with st.expander("🔍 SQL Query Generated"):
                st.code(result["sql_used"], language="sql")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "route": result["route"],
        "sql": result["sql_used"]
    })

    st.rerun()

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    process_question(question)

if question := st.chat_input("Ask anything about your patients..."):
    process_question(question)