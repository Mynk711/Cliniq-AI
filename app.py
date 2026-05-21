import warnings
warnings.filterwarnings("ignore")
import time
import streamlit as st
from engine import answer, classify_query
from database import init_db
from document import build_index
import os

# Page config
st.set_page_config(
    page_title="CliniqAI",
    page_icon="⚕️",
    layout="wide"
)

# Initialize on first run
@st.cache_resource
@st.cache_resource
def initialize():
    import os
    db_path = os.path.join(os.path.dirname(__file__), "cliniqai.db")
    index_path = os.path.join(os.path.dirname(__file__), "cliniqai.index")
    
    if not os.path.exists(db_path):
        init_db()
    if not os.path.exists(index_path):
        build_index()
    return True

initialize()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚕️ CliniqAI")
    st.caption("Intelligent Clinical Assistant: Powered by MIMIC-III & Groq")
with col2:
    st.metric("Patients", "100")
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
    "S": {"label": "SQL", "color": "#1f77b4", "icon": "🗄️"},
    "R": {"label": "RAG", "color": "#2ca02c", "icon": "🧠"},
    "H": {"label": "HYBRID", "color": "#9467bd", "icon": "⚡"},
    "D": {"label": "DIRECT", "color": "#7f7f7f", "icon": "💬"},
}

def render_route_badge(route):
    config = ROUTE_CONFIG.get(route, ROUTE_CONFIG["D"])
    st.markdown(
        f"""<span style='
            background-color: {config["color"]}22;
            color: {config["color"]};
            border: 1px solid {config["color"]};
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        '>{config["icon"]} {config["label"]}</span>""",
        unsafe_allow_html=True
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display chat history
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
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Thinking states
        status = st.empty()

        status.markdown("🔄 **Classifying intent...**")
        time.sleep(0.3)

        route = classify_query(question)
        config = ROUTE_CONFIG.get(route.upper(), ROUTE_CONFIG["D"])

        if route == "s":
            status.markdown(f"🗄️ **Querying SQL database...**")
        elif route == "r":
            status.markdown(f"🧠 **Searching FAISS vector space...**")
        elif route == "h":
            status.markdown(f"⚡ **Running parallel SQL + RAG retrieval...**")
        else:
            status.markdown(f"💬 **Thinking...**")

        chat_history = build_chat_history()
        result = answer(question, chat_history)

        status.markdown("✍️ **Synthesizing clinical answer...**")
        time.sleep(0.3)
        status.empty()

        # Render answer
        st.markdown(result["answer"])
        render_route_badge(result["route"])

        if result["sql_used"]:
            with st.expander("🔍 SQL Query Generated"):
                st.code(result["sql_used"], language="sql")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "route": result["route"],
        "sql": result["sql_used"]
    })

    st.rerun()

# Handle sidebar button clicks
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    process_question(question)

# Chat input
if question := st.chat_input("Ask anything about your patients..."):
    process_question(question)

# import warnings
# warnings.filterwarnings("ignore")
# import time
# import streamlit as st
# from engine import answer, classify_query
# from database import init_db
# from document import build_index
# import os

# st.set_page_config(
#     page_title="CliniqAI",
#     page_icon="⚕️",
#     layout="wide"
# )

# st.markdown("""
# <style>
# .stApp { background-color: #07132A; }
# [data-testid="stSidebar"] { background-color: #0D2040; border-right: 1px solid #183560; }
# [data-testid="stSidebar"] * { color: #EFF6FF !important; }
# [data-testid="stSidebar"] .stButton > button { background-color: #07132A; color: #6B9CC4 !important; border: 1px solid #183560; border-radius: 6px; font-size: 12px; }
# [data-testid="stSidebar"] .stButton > button:hover { border-color: #1AC8D4; color: #1AC8D4 !important; }
# [data-testid="stChatMessage"] { background-color: #0D2040; border: 1px solid #183560; border-radius: 10px; margin-bottom: 8px; }
# [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: #EFF6FF !important; }
# [data-testid="stChatInput"] { background-color: #0D2040 !important; border: 1px solid #1AC8D4 !important; border-radius: 10px !important; }
# [data-testid="stChatInput"] textarea { color: #EFF6FF !important; background-color: #0D2040 !important; }
# h1, h2, h3, p, span, label { color: #EFF6FF !important; }
# [data-testid="stMetricValue"] { color: #1AC8D4 !important; }
# hr { border-color: #183560 !important; }
# .stCaption, small { color: #6B9CC4 !important; }
# #scroll-btn { position: fixed; bottom: 90px; right: 24px; width: 42px; height: 42px; background-color: #1AC8D4; color: #07132A; border: none; border-radius: 50%; font-size: 20px; cursor: pointer; z-index: 9999; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 10px rgba(26,200,212,0.4); text-decoration: none; }
# #scroll-btn:hover { background-color: #7DE8ED; }
# </style>
# <a id="scroll-btn" onclick="window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})" title="Scroll to latest">↓</a>
# """, unsafe_allow_html=True)



# @st.cache_resource
# def initialize():
#     db_path = os.path.join(os.path.dirname(__file__), "cliniqai.db")
#     index_path = os.path.join(os.path.dirname(__file__), "cliniqai.index")
#     if not os.path.exists(db_path):
#         init_db()
#     if not os.path.exists(index_path):
#         build_index()
#     return True

# initialize()

# col1, col2 = st.columns([3, 1])
# with col1:
#     st.title("⚕️ CliniqAI")
#     st.caption("Intelligent Clinical Assistant: Powered by MIMIC-III & Groq")
# with col2:
#     st.metric("Patients", "100")
#     st.metric("Data Source", "MIMIC-III")

# st.divider()

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

# ROUTE_CONFIG = {
#     "S": {"label": "SQL",    "color": "#0EA5E9", "icon": "🗄️"},
#     "R": {"label": "RAG",    "color": "#1AC8D4", "icon": "🧠"},
#     "H": {"label": "HYBRID", "color": "#6366F1", "icon": "⚡"},
#     "D": {"label": "DIRECT", "color": "#6B9CC4", "icon": "💬"},
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

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "pending_question" not in st.session_state:
#     st.session_state.pending_question = None

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
#     st.session_state.messages.append({"role": "user", "content": question})

#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         status = st.empty()
#         status.markdown("🔄 **Classifying intent...**")
#         time.sleep(0.3)

#         route = classify_query(question)

#         if route == "s":
#             status.markdown("🗄️ **Querying SQL database...**")
#         elif route == "r":
#             status.markdown("🧠 **Searching FAISS vector space...**")
#         elif route == "h":
#             status.markdown("⚡ **Running parallel SQL + RAG retrieval...**")
#         else:
#             status.markdown("💬 **Thinking...**")

#         chat_history = build_chat_history()
#         result = answer(question, chat_history)

#         status.markdown("✍️ **Synthesizing clinical answer...**")
#         time.sleep(0.3)
#         status.empty()

#         st.markdown(result["answer"])
#         render_route_badge(result["route"])

#         if result["sql_used"]:
#             with st.expander("🔍 SQL Query Generated"):
#                 st.code(result["sql_used"], language="sql")

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "route": result["route"],
#         "sql": result["sql_used"]
#     })

#     st.rerun()

# if st.session_state.pending_question:
#     question = st.session_state.pending_question
#     st.session_state.pending_question = None
#     process_question(question)

# if question := st.chat_input("Ask anything about your patients..."):
#     process_question(question)